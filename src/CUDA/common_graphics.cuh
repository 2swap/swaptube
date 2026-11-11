#include <cuda_runtime.h>
#include "color.cuh"
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/CameraProjection.h"

namespace Cuda {

__device__ __forceinline__ void depth_plot(const Cuda::ivec2& pix, float z, int col, float opacity, unsigned long long* depth, bool translucent, unsigned int* pixels, const Cuda::ivec2& wh) {
    if (pix.x < 0 || pix.x >= wh.x || pix.y < 0 || pix.y >= wh.y || z <= 0) return;
    const unsigned int a = (unsigned int)(Cuda::clamp(opacity, 0.0f, 1.0f) * ((unsigned int)col >> 24));
    if ((a < 255) != translucent) return;
    const int idx = pix.y * wh.x + pix.x;
    const unsigned int c = (a << 24) | ((unsigned int)col & 0x00ffffff);
    if (!translucent) { atomicMin(&depth[idx], ((unsigned long long)__float_as_uint(z) << 32) | c); return; }
    if (__float_as_uint(z) >= (unsigned int)(depth[idx] >> 32)) return;
    unsigned int old_pixel = pixels[idx], prev;
    while ((prev = atomicCAS(&pixels[idx], old_pixel, Cuda::color_combine(old_pixel, c))) != old_pixel) old_pixel = prev;
}

// Fill a circle on a pixel buffer
__device__ __forceinline__ void d_fill_circle(const Cuda::vec2& center, float r, int col, unsigned int* pixels, const Cuda::ivec2& wh, float opa=1.0f, bool brightest_wins = false, unsigned long long* depth = nullptr, float z = 0, float world_r = 0, bool translucent = false) { // HACK. brightest_wins, revert this
    // breakout if outside of screen
    if (center.x + r < 0 || center.x - r >= wh.x || center.y + r < 0 || center.y - r >= wh.y)
        return;
    float r2 = r*r;
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            if (sdx + dy*dy < r2) {
                if (depth) depth_plot(Cuda::ivec2(center.x + dx, center.y + dy), z - world_r * sqrtf(1 - (sdx + dy*dy) / r2), col, opa, depth, translucent, pixels, wh);
                else if (brightest_wins) atomic_overlay_pixel_brightest(Cuda::ivec2(center.x + dx, center.y + dy), col, opa, pixels, wh);
                else                atomic_overlay_pixel(Cuda::ivec2(center.x + dx, center.y + dy), col, opa, pixels, wh);
            }
        }
    }
}

__device__ __forceinline__ void bresenham(const Cuda::vec2& start, const Cuda::vec2& end, int col, float opacity, int thickness, unsigned int* pixels, const Cuda::ivec2& wh, bool is_dashed, bool brightest_wins = false, unsigned long long* depth = nullptr, float z1 = 1, float z2 = 1, bool translucent = false) { // HACK. brightest_wins, revert this
    int x1 = start.x, y1 = start.y;
    const int x2 = end.x, y2 = end.y;
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    if (dx > 10000 || dy > 10000) return;
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    int dash_counter = 0;
    int dash_counter_modulus = (wh.x + wh.y) >> 7; // Naive average
    const float steps = fmaxf(1.0f, (float)max(dx, dy));
    auto plot = [&](int x, int y) {
        if (depth) {
            const float t = dash_counter / steps;
            depth_plot(Cuda::ivec2(x, y), 1.0f / ((1 - t) / z1 + t / z2), col, opacity, depth, translucent, pixels, wh);
        }
        else if (brightest_wins) atomic_overlay_pixel_brightest(Cuda::ivec2(x, y), col, opacity, pixels, wh);
        else                atomic_overlay_pixel(Cuda::ivec2(x, y), col, opacity, pixels, wh);
    };

    while (true) {
        if(!is_dashed || ((dash_counter / dash_counter_modulus) % 2 == 0)) {
            plot(x1, y1);
            for (int i = 1; i < thickness; i++) {
                plot(x1 + i, y1);
                plot(x1 - i, y1);
                plot(x1, y1 + i);
                plot(x1, y1 - i);
            }
        }
        dash_counter++;
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 <  dx) { err += dx; y1 += sy; }
    }
}

__device__ __forceinline__ Cuda::vec3 get_raymarch_vector(
    const ivec2& pixel,
    const ivec2& wh,
    const float fov,
    const quat& camera_orientation)
{
    float scale = 1 / (sqrtf(wh.x * wh.y) * fov);
    vec3 rotated;
    rotated.x = (pixel.x - wh.x * 0.5f) * scale;
    rotated.y = -(pixel.y - wh.y * 0.5f) * scale;
    rotated.z = 1.0f;
    return rotate_vector(rotated, camera_orientation);
}

}
