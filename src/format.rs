use objc2_metal::MTLPixelFormat;
use wgpu::TextureFormat;

/// Map a wgpu [`TextureFormat`] to a Metal pixel format for common Solari / HDR formats.
pub(crate) fn texture_format_to_metal(format: TextureFormat) -> Option<MTLPixelFormat> {
    use TextureFormat as Tf;
    Some(match format {
        Tf::R8Unorm => MTLPixelFormat::R8Unorm,
        Tf::R16Float => MTLPixelFormat::R16Float,
        Tf::R32Float => MTLPixelFormat::R32Float,
        Tf::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Tf::Rg16Float => MTLPixelFormat::RG16Float,
        Tf::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Tf::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Tf::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Tf::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Tf::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Tf::Rgba32Float => MTLPixelFormat::RGBA32Float,
        Tf::Depth16Unorm => MTLPixelFormat::Depth16Unorm,
        Tf::Depth32Float => MTLPixelFormat::Depth32Float,
        Tf::Depth32FloatStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
        _ => return None,
    })
}
