use wgpu::{Device, hal::api::Metal};

/// Which MetalFX features are supported on the current device.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeatureSupport {
    /// [`MTLFXTemporalDenoisedScaler`](objc2_metal_fx::MTLFXTemporalDenoisedScaler) is available.
    pub temporal_denoised_scaler_supported: bool,
    /// Smallest supported `output / input` scale. Native denoising is `1.0`.
    ///
    /// Only meaningful when [`Self::temporal_denoised_scaler_supported`] is true.
    pub min_input_content_scale: f32,
    /// Largest supported `output / input` scale.
    ///
    /// Only meaningful when [`Self::temporal_denoised_scaler_supported`] is true.
    pub max_input_content_scale: f32,
}

/// Query MetalFX feature support for a wgpu [`Device`].
///
/// Soft-fails (leaves flags false) when the device is not Metal-backed.
pub fn query_feature_support(device: &Device, feature_support: &mut FeatureSupport) {
    *feature_support = FeatureSupport::default();

    let Some(hal_device) = (unsafe { device.as_hal::<Metal>() }) else {
        return;
    };

    let raw_device = hal_device.raw_device();
    objc2::rc::autoreleasepool(|_| {
        let supported = unsafe {
            objc2_metal_fx::MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(raw_device)
        };
        feature_support.temporal_denoised_scaler_supported = supported;
        if supported {
            unsafe {
                feature_support.min_input_content_scale =
                    objc2_metal_fx::MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(
                        raw_device,
                    );
                feature_support.max_input_content_scale =
                    objc2_metal_fx::MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(
                        raw_device,
                    );
            }
        }
    });
}
