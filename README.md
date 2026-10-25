# metalfx_wgpu - MetalFX for wgpu

A wrapper for using [MetalFX](https://developer.apple.com/documentation/metalfx) with [wgpu](https://github.com/gfx-rs/wgpu) when targeting Metal on Apple platforms.

This crate exposes **MetalFX Temporal Denoised Scaler** (`MTLFXTemporalDenoisedScaler`): neural denoising and upscaling, the Metal counterpart to DLSS Ray Reconstruction.

MetalFX ships with the operating system. There is no SDK to download or redistribute.

## Version Chart

| metalfx_wgpu | wgpu |
| :----------: | :--: |
|    0.1.0     | v30  |

## Requirements

* Apple Silicon Mac on macOS 26 or later, with `MTLFXTemporalDenoisedScaler` supported on the GPU
* wgpu 30, Metal backend
* A wgpu `Device` created for that backend

On non-Apple targets the crate compiles and exports no API. The `mock` feature does the same on Apple, for CI.

## Features

| Cargo feature | Purpose |
| --- | --- |
| `mock` | Export no API. Same idea as `dlss_wgpu`'s `mock`. |

## Usage

```rust
use metalfx_wgpu::{
    FeatureSupport, MetalFxFeatureFlags, MetalFxQualityMode, MetalFxTemporalDenoisedScaler,
    MetalFxTemporalDenoisedScalerFormats, query_feature_support,
};

let mut feature_support = FeatureSupport::default();
query_feature_support(&device, &mut feature_support);
if !feature_support.temporal_denoised_scaler_supported {
    return;
}

let upscaled_resolution = [1920, 1080];
let render_resolution = MetalFxQualityMode::Quality.render_resolution(upscaled_resolution);

let mut context = MetalFxTemporalDenoisedScaler::new(
    render_resolution,
    upscaled_resolution,
    formats,
    MetalFxFeatureFlags::AUTO_EXPOSURE | MetalFxFeatureFlags::DEPTH_REVERSED,
    &device,
)?;

// Create input and output textures with at least `context.texture_usages()`.
// Then, each frame:
let metal_fx_cb = context.render(render_parameters, &mut command_encoder)?;
queue.submit([command_encoder.finish(), metal_fx_cb]);
```

`new` is expensive. Keep the scaler and recreate it when the resolution or flags change.

Barriers are recorded on the caller's encoder. MetalFX work is returned as a separate [`CommandBuffer`](https://docs.rs/wgpu/30/wgpu/struct.CommandBuffer.html). Submit that buffer in the same `queue.submit` as `command_encoder.finish()`, immediately afterwards. wgpu forbids mixing its encoder API with raw Metal encoding on the same encoder.

## Quality modes

`MetalFxQualityMode::render_resolution` picks the input size from the output size:

| Mode | Input size | Upscale (`output / input`) |
| --- | --- | --- |
| `Auto`, `Balanced` | 1/2 | 2× |
| `Native` | 1/1 | 1× |
| `Quality` | 2/3 | 1.5× |
| `Performance` | 1/3 | 3× |

`new` rejects a scaler whose scale is outside the device range. After a successful support query, `FeatureSupport::min_input_content_scale` and `max_input_content_scale` are that range (`output / input`). Native denoising is `1.0`. A Performance preset fails on a device whose maximum is 2×.

## Texture sizes

MetalFX requires every input to be exactly the scaler's input size, and the output to be exactly the upscaled size.

Inputs that are larger in both dimensions are cropped from the top-left into a private scratch texture before encode. That covers a render target that is full output size while the scene was drawn into the top-left at input size. Inputs smaller than the input size return `MetalFxError::TextureMismatch`.

The output must already be the upscaled size, in private storage. wgpu render attachments on Metal are private.

Formats have to match the formats passed to `new`, and each texture's usage has to include what `texture_usages` reports for that slot.

## Jitter

`suggested_jitter` and `suggested_mip_bias` follow the same Halton pattern as DLSS. `suggested_jitter` returns a pixel offset.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
