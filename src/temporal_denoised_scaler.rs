use crate::{
    format::texture_format_to_metal,
    hal::{halton_sequence, metal_usage_to_wgpu, retained_texture_from_view},
};
use bitflags::bitflags;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
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
}

impl MetalFxTemporalDenoisedScaler {
    /// Create a new scaler context.
    ///
    /// This is an expensive operation. Cache the result and only recreate when settings change.
    ///
    /// Only call this when [`crate::FeatureSupport::temporal_denoised_scaler_supported`] is true.
    pub fn new(
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
        })
    }

    /// Encode MetalFX denoising/upscaling.
    ///
    /// Barriers are recorded on `command_encoder` via the wgpu API. MetalFX itself is encoded on a
    /// **separate** command encoder (raw Metal HAL only), matching `dlss_wgpu`.
    ///
    /// Submit the returned [`CommandBuffer`] in the same `queue.submit` as `command_encoder.finish()`,
    /// ordered immediately afterwards.
    pub fn render(
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
            self.scaler
                .setReactiveMaskTexture(reactive_mask.as_deref());
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
            self.scaler
                .setShouldResetHistory(render_parameters.reset);
            self.scaler.setDepthReversed(
                self.feature_flags
                    .contains(MetalFxFeatureFlags::DEPTH_REVERSED),
            );
            self.scaler.setPreExposure(render_parameters.pre_exposure);
        }

        // Encode MetalFX on a dedicated encoder so we never mix wgpu + raw HAL APIs
        // on the caller's encoder (wgpu panics if you do).
        let mut metal_fx_encoder =
            self.device
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
                self.scaler.encodeToCommandBuffer(command_buffer);
                Ok(())
            })?;
        }

        // Keep retained textures alive until MetalFX has been encoded.
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
        ));

        Ok(metal_fx_encoder.finish())
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

fn map_format(format: TextureFormat) -> Result<objc2_metal::MTLPixelFormat, MetalFxError> {
    texture_format_to_metal(format).ok_or(MetalFxError::UnsupportedFormat(format))
}
