# metalfx_wgpu - MetalFX for wgpu

A wrapper for using [MetalFX](https://developer.apple.com/documentation/metalfx) with [wgpu](https://github.com/gfx-rs/wgpu) when targeting Metal on Apple platforms.

Currently exposes **MetalFX Temporal Denoised Scaler** (ray reconstruction / neural denoise + upscale), the Metal counterpart to DLSS Ray Reconstruction used by Bevy Solari.

## Requirements

* macOS **26+** (or iOS 26+ / corresponding SDK) with Apple Silicon supporting `MTLFXTemporalDenoisedScaler`
* wgpu **30** Metal backend
* Xcode / macOS SDK with `MetalFX.framework`

## Features

| Cargo feature | Purpose |
|---|---|
| `mock` | Export no API (for CI). Non-Apple targets also export no API without `mock`. |

## Usage sketch

```rust
use metalfx_wgpu::{
    FeatureSupport, MetalFxFeatureFlags, MetalFxTemporalDenoisedScaler,
    MetalFxTemporalDenoisedScalerFormats, query_feature_support,
};

let mut feature_support = FeatureSupport::default();
query_feature_support(&device, &mut feature_support);
assert!(feature_support.temporal_denoised_scaler_supported);

let mut context = MetalFxTemporalDenoisedScaler::new(
    render_resolution,
    upscaled_resolution,
    formats,
    MetalFxFeatureFlags::AUTO_EXPOSURE | MetalFxFeatureFlags::DEPTH_REVERSED,
    &device,
)?;

let metal_fx_cb = context.render(render_parameters, &mut command_encoder)?;
queue.submit([command_encoder.finish(), metal_fx_cb]);
```

Like `dlss_wgpu`, barriers are recorded on the caller's encoder and MetalFX work is returned as a separate `CommandBuffer` to submit immediately afterwards (wgpu forbids mixing its encoder API with raw HAL encoding on the same encoder).

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
