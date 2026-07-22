use wgpu::{Device, hal::api::Metal};

/// Which MetalFX features are supported on the current device.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeatureSupport {
    /// [`MTLFXTemporalDenoisedScaler`](objc2_metal_fx::MTLFXTemporalDenoisedScaler) is available.
    pub temporal_denoised_scaler_supported: bool,
}

/// Query MetalFX feature support for a wgpu [`Device`].
///
/// Soft-fails (leaves flags false) when the device is not Metal-backed.
pub fn query_feature_support(device: &Device, feature_support: &mut FeatureSupport) {
    *feature_support = FeatureSupport::default();

    let Some(hal_device) = (unsafe { device.as_hal::<Metal>() }) else {
        return;
    };

    let supported = unsafe {
        objc2_metal_fx::MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(
            hal_device.raw_device(),
        )
    };
    feature_support.temporal_denoised_scaler_supported = supported;
}
