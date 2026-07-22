//! # metalfx_wgpu
//!
//! Rust bindings for integrating Apple [MetalFX](https://developer.apple.com/documentation/metalfx)
//! with the [`wgpu`] graphics API.
//!
//! This crate only works with wgpu's Metal backend on Apple platforms.
//! On other targets, or with the `mock` feature, the crate exports no API
//! (same pattern as [`dlss_wgpu`](https://crates.io/crates/dlss_wgpu) + `mock`).
//!
//! ## API Usage
//! ```compile_fail
//! use metalfx_wgpu::{
//!     FeatureSupport, MetalFxFeatureFlags, MetalFxTemporalDenoisedScaler,
//!     MetalFxTemporalDenoisedScalerFormats, MetalFxTemporalDenoisedScalerRenderParameters,
//! };
//!
//! let mut feature_support = FeatureSupport::default();
//! metalfx_wgpu::query_feature_support(&device, &mut feature_support);
//! if !feature_support.temporal_denoised_scaler_supported {
//!     return;
//! }
//!
//! let mut context = MetalFxTemporalDenoisedScaler::new(
//!     render_resolution,
//!     upscaled_resolution,
//!     formats,
//!     MetalFxFeatureFlags::AUTO_EXPOSURE | MetalFxFeatureFlags::DEPTH_REVERSED,
//!     &device,
//! )?;
//!
//! let metal_fx_cb = context.render(render_parameters, &mut command_encoder)?;
//! queue.submit([command_encoder.finish(), metal_fx_cb]);
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(not(feature = "mock"), target_vendor = "apple"))]
mod feature_support;
#[cfg(all(not(feature = "mock"), target_vendor = "apple"))]
mod format;
#[cfg(all(not(feature = "mock"), target_vendor = "apple"))]
mod hal;
#[cfg(all(not(feature = "mock"), target_vendor = "apple"))]
mod temporal_denoised_scaler;

#[cfg(all(not(feature = "mock"), target_vendor = "apple"))]
pub use feature_support::{FeatureSupport, query_feature_support};
#[cfg(all(not(feature = "mock"), target_vendor = "apple"))]
pub use temporal_denoised_scaler::{
    MetalFxError, MetalFxFeatureFlags, MetalFxQualityMode, MetalFxTemporalDenoisedScaler,
    MetalFxTemporalDenoisedScalerFormats, MetalFxTemporalDenoisedScalerRenderParameters,
    MetalFxTextureUsages,
};
