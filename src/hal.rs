use crate::MetalFxError;
use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLTexture, MTLTextureUsage};
use wgpu::{Texture, TextureUsages, TextureView, hal::api::Metal};

pub(crate) fn retained_texture_from_view(
    texture_view: &TextureView,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, MetalFxError> {
    retained_texture(texture_view.texture())
}

pub(crate) fn retained_texture(
    texture: &Texture,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, MetalFxError> {
    let Some(hal_texture) = (unsafe { texture.as_hal::<Metal>() }) else {
        return Err(MetalFxError::NotMetalBackend);
    };
    Ok(hal_texture.raw_handle().retain())
}

pub(crate) fn metal_usage_to_wgpu(usage: MTLTextureUsage) -> TextureUsages {
    let mut result = TextureUsages::empty();
    if usage.contains(MTLTextureUsage::ShaderRead) {
        result |= TextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(MTLTextureUsage::ShaderWrite) {
        result |= TextureUsages::STORAGE_BINDING;
    }
    if usage.contains(MTLTextureUsage::RenderTarget) {
        result |= TextureUsages::RENDER_ATTACHMENT;
    }
    result
}

pub(crate) fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index = (index as f32 / base as f32).floor() as u32;
    }
    result
}
