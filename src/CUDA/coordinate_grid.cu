#include "../Host_Device_Shared/vec.h"
#include <cstdint>
#include "../Host_Device_Shared/helpers.h"
#include "color.cuh"

constexpr float HALF_GRID_LINE_THICKNESS = 0.06f;

__device__ __forceinline__ float is_near_integer(float val, float frac_part) {
    float half_thickness = HALF_GRID_LINE_THICKNESS * exp2f(-frac_part);
    float closeness = half_thickness - fabsf(val - roundf(val));
    return closeness > 0 ? closeness / half_thickness : 0;
}

__device__ __forceinline__ uint32_t coordinate_line(float x, float y, float zoom) {
    int floor_zoom = floorf(zoom);
    float exp_zoom = floor_zoom >= 0 ? (1 << floor_zoom) : (1.0f / (1 << -floor_zoom));
    float frac_part = zoom - floor_zoom;
    float close1 = max(is_near_integer(x * exp_zoom      , frac_part), is_near_integer(y * exp_zoom      , frac_part));
    float close2 = max(is_near_integer(x * exp_zoom + 0.5, frac_part), is_near_integer(y * exp_zoom + 0.5, frac_part));
    float opacity_f = max(close1, close2 * frac_part);
    if (opacity_f > 0) {
        int opacity = (int)(opacity_f * 255);
        return opacity << 24 | 0x004400ff;
    }
    return 0;
}

__global__ void render_coordinate_grid_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by)
{
    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;
    Cuda::vec2 point_vec = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);

    float zoom = log2f(rx_by.x - lx_ty.x);
    uint32_t color = coordinate_line(point_vec.x, point_vec.y, -zoom + 5);
    if(color == 0) return;

    // Effect is stronger at center of screen
    Cuda::vec2 screen_center = (lx_ty + rx_by) * 0.5f;
    Cuda::vec2 to_center = (point_vec - screen_center) / ((rx_by - lx_ty) * 0.5f);
    float roundrect = to_center.x * to_center.x * to_center.x * to_center.x + to_center.y * to_center.y * to_center.y * to_center.y;
    float opacity = Cuda::lerp(1, .2, roundrect);
    overlay_pixel(pixel, color, opacity, pixels, wh);
}

extern "C" void draw_coordinate_grid(uint32_t* d_pixels, const Cuda::ivec2& pix_wh, const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by)
{
    dim3 blockSize(16, 16);
    dim3 numBlocks((pix_wh.x + blockSize.x - 1) / blockSize.x, (pix_wh.y + blockSize.y - 1) / blockSize.y);
    render_coordinate_grid_kernel<<<numBlocks, blockSize>>>(d_pixels, pix_wh, lx_ty, rx_by);
    cudaDeviceSynchronize();
}
