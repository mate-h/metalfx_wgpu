use crate::{
    format::texture_format_to_metal,
    hal::{halton_sequence, metal_usage_to_wgpu, retained_texture_from_view},
};
use bitflags::bitflags;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLOrigin,
    MTLPixelFormat, MTLResource, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};
use objc2_metal_fx::{
    MTLFXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerBase,
    MTLFXTemporalDenoisedScalerDescriptor,
};
use std::iter;
use thiserror::Error;
use wgpu::{
    CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, Texture, TextureFormat,
    TextureTransition, TextureUses, TextureView, hal::api::Metal,
};

/// Required guide inputs plus the optional textures [`MetalFxTemporalDenoisedScaler::render`] may crop.
const INPUT_SLOT_COUNT: usize = 12;
/// Preset math rounds each axis, so scale checks allow a little slack around the device limits.
const SCALE_EPSILON: f32 = 1e-3;

type MetalTexture = Retained<ProtocolObject<dyn MTLTexture>>;

/// Upscaling quality preset controlling render vs output resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalFxQualityMode {
    /// Approximately 1/2 render resolution (2x upscale).
    #[default]
    Auto,
    /// No upscaling; denoise only.
    Native,
    /// Approximately 2/3 render resolution.
    Quality,
    /// Approximately 1/2 render resolution.
    Balanced,
    /// Approximately 1/3 render resolution.
    Performance,
}

impl MetalFxQualityMode {
    /// Compute the pre-upscale render resolution for a given output resolution.
    pub fn render_resolution(self, upscaled_resolution: [u32; 2]) -> [u32; 2] {
        let scale = match self {
            Self::Auto | Self::Balanced => 0.5,
            Self::Native => 1.0,
            Self::Quality => 2.0 / 3.0,
            Self::Performance => 1.0 / 3.0,
        };
        [
            ((upscaled_resolution[0] as f32) * scale).round().max(1.0) as u32,
            ((upscaled_resolution[1] as f32) * scale).round().max(1.0) as u32,
        ]
    }
}

bitflags! {
    /// Configuration flags for [`MetalFxTemporalDenoisedScaler`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MetalFxFeatureFlags: u32 {
        /// Automatically calculate exposure each frame.
        const AUTO_EXPOSURE = 1 << 0;
        /// Depth buffer uses zero for the farthest distance (reversed-Z).
        const DEPTH_REVERSED = 1 << 1;
        /// Enable the specular hit-distance guide texture.
        const SPECULAR_HIT_DISTANCE = 1 << 2;
        /// Enable the reactive mask texture.
        const REACTIVE_MASK = 1 << 3;
        /// Enable the denoise strength mask texture.
        const DENOISE_STRENGTH_MASK = 1 << 4;
        /// Enable the transparency overlay texture.
        const TRANSPARENCY_OVERLAY = 1 << 5;
        /// Compile the fastest upscaler synchronously at creation time.
        const REQUIRES_SYNCHRONOUS_INIT = 1 << 6;
    }
}

impl Default for MetalFxFeatureFlags {
    fn default() -> Self {
        Self::AUTO_EXPOSURE | Self::DEPTH_REVERSED
    }
}

/// Errors from MetalFX operations.
#[derive(Debug, Error)]
pub enum MetalFxError {
    #[error("wgpu device is not using the Metal backend")]
    NotMetalBackend,
    #[error("MetalFX temporal denoised scaler is not supported on this device")]
    Unsupported,
    #[error("failed to create MetalFX temporal denoised scaler")]
    CreationFailed,
    #[error("unsupported texture format: {0:?}")]
    UnsupportedFormat(TextureFormat),
    #[error("MetalFX command buffer is unavailable on the wgpu encoder")]
    MissingCommandBuffer,
    #[error("failed to create a Metal blit encoder")]
    MissingBlitEncoder,
    /// `scale` is `output / input` on one axis.
    #[error("input content scale {scale} is outside the supported range {min}..={max}")]
    UnsupportedScale { scale: f32, min: f32, max: f32 },
    #[error("{texture} texture: {reason}")]
    TextureMismatch {
        texture: &'static str,
        reason: String,
    },
}

/// Pixel formats for textures consumed/produced by [`MetalFxTemporalDenoisedScaler`].
#[derive(Debug, Clone, Copy)]
pub struct MetalFxTemporalDenoisedScalerFormats {
    pub color: TextureFormat,
    pub depth: TextureFormat,
    pub motion_vectors: TextureFormat,
    pub diffuse_albedo: TextureFormat,
    pub specular_albedo: TextureFormat,
    pub normals: TextureFormat,
    pub roughness: TextureFormat,
    pub output: TextureFormat,
    pub specular_hit_distance: Option<TextureFormat>,
    pub reactive_mask: Option<TextureFormat>,
    pub denoise_strength_mask: Option<TextureFormat>,
    pub transparency_overlay: Option<TextureFormat>,
}

/// Minimum wgpu texture usages required by a created scaler instance.
#[derive(Debug, Clone, Copy)]
pub struct MetalFxTextureUsages {
    pub color: wgpu::TextureUsages,
    pub depth: wgpu::TextureUsages,
    pub motion_vectors: wgpu::TextureUsages,
    pub diffuse_albedo: wgpu::TextureUsages,
    pub specular_albedo: wgpu::TextureUsages,
    pub normals: wgpu::TextureUsages,
    pub roughness: wgpu::TextureUsages,
    pub output: wgpu::TextureUsages,
}

/// Camera-specific object for MetalFX temporal denoised scaling (ray reconstruction).
pub struct MetalFxTemporalDenoisedScaler {
    scaler: Retained<ProtocolObject<dyn MTLFXTemporalDenoisedScaler>>,
    device: Device,
    render_resolution: [u32; 2],
    upscaled_resolution: [u32; 2],
    feature_flags: MetalFxFeatureFlags,
    texture_usages: MetalFxTextureUsages,
    /// Exact-size copies of inputs that were larger than [`Self::render_resolution`].
    input_scratch: [Option<MetalTexture>; INPUT_SLOT_COUNT],
}

impl MetalFxTemporalDenoisedScaler {
    /// Create a new scaler context.
    ///
    /// This is an expensive operation. Cache the result and only recreate when settings change.
    ///
    /// Only call this when [`crate::FeatureSupport::temporal_denoised_scaler_supported`] is true.
    ///
    /// Returns [`MetalFxError::UnsupportedScale`] when either axis of `upscaled / render` is
    /// outside the device's input-content scale range.
    pub fn new(
        render_resolution: [u32; 2],
        upscaled_resolution: [u32; 2],
        formats: MetalFxTemporalDenoisedScalerFormats,
        feature_flags: MetalFxFeatureFlags,
        device: &Device,
    ) -> Result<Self, MetalFxError> {
        // MetalFX autoreleases. Bevy's render thread has no pool, so drain one here.
        autoreleasepool(|_| {
            Self::new_inner(
                render_resolution,
                upscaled_resolution,
                formats,
                feature_flags,
                device,
            )
        })
    }

    fn new_inner(
        render_resolution: [u32; 2],
        upscaled_resolution: [u32; 2],
        formats: MetalFxTemporalDenoisedScalerFormats,
        feature_flags: MetalFxFeatureFlags,
        device: &Device,
    ) -> Result<Self, MetalFxError> {
        let Some(hal_device) = (unsafe { device.as_hal::<Metal>() }) else {
            return Err(MetalFxError::NotMetalBackend);
        };
        let metal_device = hal_device.raw_device();

        if !unsafe { MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(metal_device) } {
            return Err(MetalFxError::Unsupported);
        }

        let (min_scale, max_scale) = unsafe {
            (
                MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(
                    metal_device,
                ),
                MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(
                    metal_device,
                ),
            )
        };
        // Preset math rounds each axis, so allow a little slack around the device limits.
        if render_resolution[0] == 0 || render_resolution[1] == 0 {
            return Err(MetalFxError::UnsupportedScale {
                scale: 0.0,
                min: min_scale,
                max: max_scale,
            });
        }
        for scale in [
            upscaled_resolution[0] as f32 / render_resolution[0] as f32,
            upscaled_resolution[1] as f32 / render_resolution[1] as f32,
        ] {
            if scale + SCALE_EPSILON < min_scale || scale - SCALE_EPSILON > max_scale {
                return Err(MetalFxError::UnsupportedScale {
                    scale,
                    min: min_scale,
                    max: max_scale,
                });
            }
        }

        let descriptor = unsafe { MTLFXTemporalDenoisedScalerDescriptor::new() };

        unsafe {
            descriptor.setColorTextureFormat(map_format(formats.color)?);
            descriptor.setDepthTextureFormat(map_format(formats.depth)?);
            descriptor.setMotionTextureFormat(map_format(formats.motion_vectors)?);
            descriptor.setDiffuseAlbedoTextureFormat(map_format(formats.diffuse_albedo)?);
            descriptor.setSpecularAlbedoTextureFormat(map_format(formats.specular_albedo)?);
            descriptor.setNormalTextureFormat(map_format(formats.normals)?);
            descriptor.setRoughnessTextureFormat(map_format(formats.roughness)?);
            descriptor.setOutputTextureFormat(map_format(formats.output)?);

            descriptor.setInputWidth(render_resolution[0] as _);
            descriptor.setInputHeight(render_resolution[1] as _);
            descriptor.setOutputWidth(upscaled_resolution[0] as _);
            descriptor.setOutputHeight(upscaled_resolution[1] as _);

            descriptor
                .setAutoExposureEnabled(feature_flags.contains(MetalFxFeatureFlags::AUTO_EXPOSURE));
            descriptor.setRequiresSynchronousInitialization(
                feature_flags.contains(MetalFxFeatureFlags::REQUIRES_SYNCHRONOUS_INIT),
            );

            let specular_hit = feature_flags.contains(MetalFxFeatureFlags::SPECULAR_HIT_DISTANCE);
            descriptor.setSpecularHitDistanceTextureEnabled(specular_hit);
            if specular_hit {
                let format = formats
                    .specular_hit_distance
                    .ok_or(MetalFxError::UnsupportedFormat(TextureFormat::R16Float))?;
                descriptor.setSpecularHitDistanceTextureFormat(map_format(format)?);
            }

            let reactive = feature_flags.contains(MetalFxFeatureFlags::REACTIVE_MASK);
            descriptor.setReactiveMaskTextureEnabled(reactive);
            if reactive {
                let format = formats
                    .reactive_mask
                    .ok_or(MetalFxError::UnsupportedFormat(TextureFormat::R8Unorm))?;
                descriptor.setReactiveMaskTextureFormat(map_format(format)?);
            }

            let denoise_strength =
                feature_flags.contains(MetalFxFeatureFlags::DENOISE_STRENGTH_MASK);
            descriptor.setDenoiseStrengthMaskTextureEnabled(denoise_strength);
            if denoise_strength {
                let format = formats
                    .denoise_strength_mask
                    .ok_or(MetalFxError::UnsupportedFormat(TextureFormat::R8Unorm))?;
                descriptor.setDenoiseStrengthMaskTextureFormat(map_format(format)?);
            }

            let transparency = feature_flags.contains(MetalFxFeatureFlags::TRANSPARENCY_OVERLAY);
            descriptor.setTransparencyOverlayTextureEnabled(transparency);
            if transparency {
                let format = formats
                    .transparency_overlay
                    .ok_or(MetalFxError::UnsupportedFormat(TextureFormat::Rgba16Float))?;
                descriptor.setTransparencyOverlayTextureFormat(map_format(format)?);
            }
        }

        let scaler = unsafe { descriptor.newTemporalDenoisedScalerWithDevice(metal_device) }
            .ok_or(MetalFxError::CreationFailed)?;

        let texture_usages = unsafe {
            MetalFxTextureUsages {
                color: metal_usage_to_wgpu(scaler.colorTextureUsage()),
                depth: metal_usage_to_wgpu(scaler.depthTextureUsage()),
                motion_vectors: metal_usage_to_wgpu(scaler.motionTextureUsage()),
                diffuse_albedo: metal_usage_to_wgpu(scaler.diffuseAlbedoTextureUsage()),
                specular_albedo: metal_usage_to_wgpu(scaler.specularAlbedoTextureUsage()),
                normals: metal_usage_to_wgpu(scaler.normalTextureUsage()),
                roughness: metal_usage_to_wgpu(scaler.roughnessTextureUsage()),
                output: metal_usage_to_wgpu(scaler.outputTextureUsage()),
            }
        };

        // Depth reversed is a runtime property (not descriptor-only).
        unsafe {
            scaler.setDepthReversed(feature_flags.contains(MetalFxFeatureFlags::DEPTH_REVERSED));
        }

        Ok(Self {
            scaler,
            device: device.clone(),
            render_resolution,
            upscaled_resolution,
            feature_flags,
            texture_usages,
            input_scratch: std::array::from_fn(|_| None),
        })
    }

    /// Encode MetalFX denoising/upscaling.
    ///
    /// Barriers are recorded on `command_encoder` via the wgpu API. MetalFX itself is encoded on a
    /// **separate** command encoder (raw Metal HAL only), matching `dlss_wgpu`.
    ///
    /// Inputs larger than [`Self::render_resolution`] are cropped from the top-left into a private
    /// scratch texture. MetalFX requires each input to be exactly that size, and the output to be
    /// exactly [`Self::upscaled_resolution`].
    ///
    /// Submit the returned [`CommandBuffer`] in the same `queue.submit` as `command_encoder.finish()`,
    /// ordered immediately afterwards.
    pub fn render(
        &mut self,
        render_parameters: MetalFxTemporalDenoisedScalerRenderParameters<'_>,
        command_encoder: &mut CommandEncoder,
    ) -> Result<CommandBuffer, MetalFxError> {
        autoreleasepool(|_| self.render_inner(render_parameters, command_encoder))
    }

    fn render_inner(
        &mut self,
        render_parameters: MetalFxTemporalDenoisedScalerRenderParameters<'_>,
        command_encoder: &mut CommandEncoder,
    ) -> Result<CommandBuffer, MetalFxError> {
        command_encoder.transition_resources(iter::empty(), render_parameters.barrier_list());

        let color = retained_texture_from_view(render_parameters.color)?;
        let depth = retained_texture_from_view(render_parameters.depth)?;
        let motion_vectors = retained_texture_from_view(render_parameters.motion_vectors)?;
        let diffuse_albedo = retained_texture_from_view(render_parameters.diffuse_albedo)?;
        let specular_albedo = retained_texture_from_view(render_parameters.specular_albedo)?;
        let normals = retained_texture_from_view(render_parameters.normals)?;
        let roughness = retained_texture_from_view(render_parameters.roughness)?;
        let output = retained_texture_from_view(render_parameters.output)?;

        let specular_hit_distance = render_parameters
            .specular_hit_distance
            .map(retained_texture_from_view)
            .transpose()?;
        let reactive_mask = render_parameters
            .reactive_mask
            .map(retained_texture_from_view)
            .transpose()?;
        let denoise_strength_mask = render_parameters
            .denoise_strength_mask
            .map(retained_texture_from_view)
            .transpose()?;
        let transparency_overlay = render_parameters
            .transparency_overlay
            .map(retained_texture_from_view)
            .transpose()?;
        let exposure = render_parameters
            .exposure
            .map(retained_texture_from_view)
            .transpose()?;

        let expected = unsafe {
            InputTextureSpec {
                color: (
                    self.scaler.colorTextureFormat(),
                    self.scaler.colorTextureUsage(),
                ),
                depth: (
                    self.scaler.depthTextureFormat(),
                    self.scaler.depthTextureUsage(),
                ),
                motion: (
                    self.scaler.motionTextureFormat(),
                    self.scaler.motionTextureUsage(),
                ),
                diffuse_albedo: (
                    self.scaler.diffuseAlbedoTextureFormat(),
                    self.scaler.diffuseAlbedoTextureUsage(),
                ),
                specular_albedo: (
                    self.scaler.specularAlbedoTextureFormat(),
                    self.scaler.specularAlbedoTextureUsage(),
                ),
                normal: (
                    self.scaler.normalTextureFormat(),
                    self.scaler.normalTextureUsage(),
                ),
                roughness: (
                    self.scaler.roughnessTextureFormat(),
                    self.scaler.roughnessTextureUsage(),
                ),
                specular_hit_distance: (
                    self.scaler.specularHitDistanceTextureFormat(),
                    self.scaler.specularHitDistanceTextureUsage(),
                ),
                reactive_mask: (
                    self.scaler.reactiveMaskTextureFormat(),
                    self.scaler.reactiveTextureUsage(),
                ),
                denoise_strength_mask: (
                    self.scaler.denoiseStrengthMaskTextureFormat(),
                    self.scaler.denoiseStrengthMaskTextureUsage(),
                ),
                transparency_overlay: (
                    self.scaler.transparencyOverlayTextureFormat(),
                    self.scaler.transparencyOverlayTextureUsage(),
                ),
                output: (
                    self.scaler.outputTextureFormat(),
                    self.scaler.outputTextureUsage(),
                ),
            }
        };

        let mut crops = Vec::new();
        let color = self.fit_input(0, "color", color, Some(expected.color), &mut crops)?;
        let depth = self.fit_input(1, "depth", depth, Some(expected.depth), &mut crops)?;
        let motion_vectors = self.fit_input(
            2,
            "motion",
            motion_vectors,
            Some(expected.motion),
            &mut crops,
        )?;
        let diffuse_albedo = self.fit_input(
            3,
            "diffuse_albedo",
            diffuse_albedo,
            Some(expected.diffuse_albedo),
            &mut crops,
        )?;
        let specular_albedo = self.fit_input(
            4,
            "specular_albedo",
            specular_albedo,
            Some(expected.specular_albedo),
            &mut crops,
        )?;
        let normals = self.fit_input(5, "normal", normals, Some(expected.normal), &mut crops)?;
        let roughness = self.fit_input(
            6,
            "roughness",
            roughness,
            Some(expected.roughness),
            &mut crops,
        )?;
        let specular_hit_distance = specular_hit_distance
            .map(|texture| {
                self.fit_input(
                    7,
                    "specular_hit_distance",
                    texture,
                    Some(expected.specular_hit_distance),
                    &mut crops,
                )
            })
            .transpose()?;
        let reactive_mask = reactive_mask
            .map(|texture| {
                self.fit_input(
                    8,
                    "reactive_mask",
                    texture,
                    Some(expected.reactive_mask),
                    &mut crops,
                )
            })
            .transpose()?;
        let denoise_strength_mask = denoise_strength_mask
            .map(|texture| {
                self.fit_input(
                    9,
                    "denoise_strength_mask",
                    texture,
                    Some(expected.denoise_strength_mask),
                    &mut crops,
                )
            })
            .transpose()?;
        let transparency_overlay = transparency_overlay
            .map(|texture| {
                self.fit_input(
                    10,
                    "transparency_overlay",
                    texture,
                    Some(expected.transparency_overlay),
                    &mut crops,
                )
            })
            .transpose()?;
        // MetalFX has no format/usage getters for the exposure texture.
        let exposure = exposure
            .map(|texture| self.fit_input(11, "exposure", texture, None, &mut crops))
            .transpose()?;

        validate_texture("output", &output, expected.output.0, expected.output.1)?;
        let output_size = [output.width() as u32, output.height() as u32];
        if output_size != self.upscaled_resolution {
            return Err(MetalFxError::TextureMismatch {
                texture: "output",
                reason: format!(
                    "size is {output_size:?}, expected {:?}",
                    self.upscaled_resolution
                ),
            });
        }
        if output.storageMode() != MTLStorageMode::Private {
            return Err(MetalFxError::TextureMismatch {
                texture: "output",
                reason: "MetalFX requires private storage".into(),
            });
        }

        let motion_scale = render_parameters.motion_vector_scale.unwrap_or([1.0, 1.0]);

        unsafe {
            self.scaler.setColorTexture(Some(&color));
            self.scaler.setDepthTexture(Some(&depth));
            self.scaler.setMotionTexture(Some(&motion_vectors));
            self.scaler.setDiffuseAlbedoTexture(Some(&diffuse_albedo));
            self.scaler.setSpecularAlbedoTexture(Some(&specular_albedo));
            self.scaler.setNormalTexture(Some(&normals));
            self.scaler.setRoughnessTexture(Some(&roughness));
            self.scaler.setOutputTexture(Some(&output));

            self.scaler
                .setSpecularHitDistanceTexture(specular_hit_distance.as_deref());
            self.scaler.setReactiveMaskTexture(reactive_mask.as_deref());
            self.scaler
                .setDenoiseStrengthMaskTexture(denoise_strength_mask.as_deref());
            self.scaler
                .setTransparencyOverlayTexture(transparency_overlay.as_deref());
            self.scaler.setExposureTexture(exposure.as_deref());

            self.scaler
                .setJitterOffsetX(render_parameters.jitter_offset[0]);
            self.scaler
                .setJitterOffsetY(render_parameters.jitter_offset[1]);
            self.scaler.setMotionVectorScaleX(motion_scale[0]);
            self.scaler.setMotionVectorScaleY(motion_scale[1]);
            self.scaler.setShouldResetHistory(render_parameters.reset);
            self.scaler.setDepthReversed(
                self.feature_flags
                    .contains(MetalFxFeatureFlags::DEPTH_REVERSED),
            );
            self.scaler.setPreExposure(render_parameters.pre_exposure);
        }

        let input_resolution = self.render_resolution;
        let scaler = &self.scaler;

        // Encode MetalFX on a dedicated encoder so we never mix wgpu + raw HAL APIs
        // on the caller's encoder (wgpu panics if you do).
        let mut metal_fx_encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("metal_fx_temporal_denoised_scaler"),
            });
        unsafe {
            metal_fx_encoder.as_hal_mut::<Metal, _, _>(|hal_encoder| {
                let Some(hal_encoder) = hal_encoder else {
                    return Err(MetalFxError::NotMetalBackend);
                };
                let Some(command_buffer) = hal_encoder.raw_command_buffer() else {
                    return Err(MetalFxError::MissingCommandBuffer);
                };
                if !crops.is_empty() {
                    let Some(blit) = command_buffer.blitCommandEncoder() else {
                        return Err(MetalFxError::MissingBlitEncoder);
                    };
                    let origin = MTLOrigin { x: 0, y: 0, z: 0 };
                    let size = MTLSize {
                        width: input_resolution[0] as usize,
                        height: input_resolution[1] as usize,
                        depth: 1,
                    };
                    for (source, scratch) in &crops {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            source,
                            0,
                            0,
                            origin,
                            size,
                            scratch,
                            0,
                            0,
                            origin,
                        );
                    }
                    blit.endEncoding();
                }
                scaler.encodeToCommandBuffer(command_buffer);
                Ok(())
            })?;
        }

        // wgpu still owns the source textures. Scratch copies stay in `input_scratch`.
        drop((
            color,
            depth,
            motion_vectors,
            diffuse_albedo,
            specular_albedo,
            normals,
            roughness,
            output,
            specular_hit_distance,
            reactive_mask,
            denoise_strength_mask,
            transparency_overlay,
            exposure,
            crops,
        ));

        Ok(metal_fx_encoder.finish())
    }

    /// Return `source` when it is already the input size. Crop the top-left when it is larger.
    fn fit_input(
        &mut self,
        slot: usize,
        name: &'static str,
        source: MetalTexture,
        expected: Option<(MTLPixelFormat, MTLTextureUsage)>,
        crops: &mut Vec<(MetalTexture, MetalTexture)>,
    ) -> Result<MetalTexture, MetalFxError> {
        if let Some((format, usage)) = expected {
            validate_texture(name, &source, format, usage)?;
        }
        let size = [source.width() as u32, source.height() as u32];
        let input = self.render_resolution;
        if size == input {
            return Ok(source);
        }
        if size[0] >= input[0] && size[1] >= input[1] {
            let (format, usage) =
                expected.unwrap_or_else(|| (source.pixelFormat(), source.usage()));
            let scratch = self.scratch_texture(slot, name, &source, format, usage)?;
            crops.push((source, scratch.clone()));
            return Ok(scratch);
        }
        Err(MetalFxError::TextureMismatch {
            texture: name,
            reason: format!("size is {size:?}, smaller than the input size {input:?}"),
        })
    }

    fn scratch_texture(
        &mut self,
        slot: usize,
        name: &'static str,
        source: &ProtocolObject<dyn MTLTexture>,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
    ) -> Result<MetalTexture, MetalFxError> {
        if let Some(existing) = &self.input_scratch[slot] {
            return Ok(existing.clone());
        }
        let [width, height] = self.render_resolution;
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(usage);
        descriptor.setStorageMode(MTLStorageMode::Private);
        let texture = source
            .device()
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| MetalFxError::TextureMismatch {
                texture: name,
                reason: "failed to allocate a scratch texture for cropping".into(),
            })?;
        self.input_scratch[slot] = Some(texture.clone());
        Ok(texture)
    }

    /// Suggested subpixel camera jitter for a given frame.
    pub fn suggested_jitter(&self, frame_number: u32, render_resolution: [u32; 2]) -> [f32; 2] {
        let ratio = self.upscaled_resolution[0] as f32 / render_resolution[0] as f32;
        let phase_count = ((8.0 * ratio * ratio) as u32).max(32);
        let i = frame_number % phase_count;
        [halton_sequence(i, 2) - 0.5, halton_sequence(i, 3) - 0.5]
    }

    /// Suggested mip bias to apply when sampling textures.
    pub fn suggested_mip_bias(&self, render_resolution: [u32; 2]) -> f32 {
        (render_resolution[0] as f32 / self.upscaled_resolution[0] as f32).log2() - 1.0
    }

    /// The upscaled resolution MetalFX will output at.
    pub fn upscaled_resolution(&self) -> [u32; 2] {
        self.upscaled_resolution
    }

    /// The resolution the camera should render at, pre-upscaling.
    pub fn render_resolution(&self) -> [u32; 2] {
        self.render_resolution
    }

    /// Feature flags used when this scaler was created.
    pub fn feature_flags(&self) -> MetalFxFeatureFlags {
        self.feature_flags
    }

    /// Minimum texture usages required by this scaler instance.
    pub fn texture_usages(&self) -> MetalFxTextureUsages {
        self.texture_usages
    }
}

unsafe impl Send for MetalFxTemporalDenoisedScaler {}
unsafe impl Sync for MetalFxTemporalDenoisedScaler {}

/// Inputs and output resources needed for rendering [`MetalFxTemporalDenoisedScaler`].
pub struct MetalFxTemporalDenoisedScalerRenderParameters<'a> {
    pub diffuse_albedo: &'a TextureView,
    pub specular_albedo: &'a TextureView,
    pub normals: &'a TextureView,
    pub roughness: &'a TextureView,
    pub color: &'a TextureView,
    pub depth: &'a TextureView,
    pub motion_vectors: &'a TextureView,
    pub output: &'a TextureView,
    pub reset: bool,
    pub jitter_offset: [f32; 2],
    pub motion_vector_scale: Option<[f32; 2]>,
    pub pre_exposure: f32,
    pub specular_hit_distance: Option<&'a TextureView>,
    pub reactive_mask: Option<&'a TextureView>,
    pub denoise_strength_mask: Option<&'a TextureView>,
    pub transparency_overlay: Option<&'a TextureView>,
    pub exposure: Option<&'a TextureView>,
}

impl<'a> MetalFxTemporalDenoisedScalerRenderParameters<'a> {
    fn barrier_list(&self) -> impl Iterator<Item = TextureTransition<&'a Texture>> {
        fn resource_barrier(texture_view: &TextureView) -> TextureTransition<&Texture> {
            TextureTransition {
                texture: texture_view.texture(),
                selector: None,
                state: TextureUses::RESOURCE,
            }
        }

        [
            Some(resource_barrier(self.diffuse_albedo)),
            Some(resource_barrier(self.specular_albedo)),
            Some(resource_barrier(self.normals)),
            Some(resource_barrier(self.roughness)),
            Some(resource_barrier(self.color)),
            Some(resource_barrier(self.depth)),
            Some(resource_barrier(self.motion_vectors)),
            self.specular_hit_distance.map(resource_barrier),
            self.reactive_mask.map(resource_barrier),
            self.denoise_strength_mask.map(resource_barrier),
            self.transparency_overlay.map(resource_barrier),
            self.exposure.map(resource_barrier),
            Some(TextureTransition {
                texture: self.output.texture(),
                selector: None,
                state: TextureUses::STORAGE_READ_WRITE,
            }),
        ]
        .into_iter()
        .flatten()
    }
}

struct InputTextureSpec {
    color: (MTLPixelFormat, MTLTextureUsage),
    depth: (MTLPixelFormat, MTLTextureUsage),
    motion: (MTLPixelFormat, MTLTextureUsage),
    diffuse_albedo: (MTLPixelFormat, MTLTextureUsage),
    specular_albedo: (MTLPixelFormat, MTLTextureUsage),
    normal: (MTLPixelFormat, MTLTextureUsage),
    roughness: (MTLPixelFormat, MTLTextureUsage),
    specular_hit_distance: (MTLPixelFormat, MTLTextureUsage),
    reactive_mask: (MTLPixelFormat, MTLTextureUsage),
    denoise_strength_mask: (MTLPixelFormat, MTLTextureUsage),
    transparency_overlay: (MTLPixelFormat, MTLTextureUsage),
    output: (MTLPixelFormat, MTLTextureUsage),
}

fn validate_texture(
    name: &'static str,
    texture: &ProtocolObject<dyn MTLTexture>,
    format: MTLPixelFormat,
    usage: MTLTextureUsage,
) -> Result<(), MetalFxError> {
    let mismatch = |reason: String| MetalFxError::TextureMismatch {
        texture: name,
        reason,
    };
    if texture.pixelFormat() != format {
        return Err(mismatch(format!(
            "pixel format is {:?}, expected {format:?}",
            texture.pixelFormat()
        )));
    }
    if !texture.usage().contains(usage) {
        return Err(mismatch(format!(
            "usage {:?} is missing {:?}",
            texture.usage(),
            usage
        )));
    }
    if texture.sampleCount() != 1 {
        return Err(mismatch("must not be multisampled".into()));
    }
    Ok(())
}

fn map_format(format: TextureFormat) -> Result<MTLPixelFormat, MetalFxError> {
    texture_format_to_metal(format).ok_or(MetalFxError::UnsupportedFormat(format))
}
