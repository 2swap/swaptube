#include <cuda_runtime.h>
#include "color.cuh"
#include "../Host_Device_Shared/vec.h"

namespace Cuda {

// Fill a circle on a pixel buffer
__device__ __forceinline__ void d_fill_circle(float cx, float cy, float r, int col, unsigned int* pixels, const Cuda::ivec2& wh, float opa=1.0f) {
    // breakout if outside of screen
    if (cx + r < 0 || cx - r >= wh.x || cy + r < 0 || cy - r >= wh.y)
        return;
    float r2 = r*r;
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            if (sdx + dy*dy < r2)
                atomic_overlay_pixel(cx + dx, cy + dy, col, opa, pixels, wh);
        }
    }
}

__device__ __forceinline__ void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness, unsigned int* pixels, const Cuda::ivec2& wh, bool is_dashed) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    if (dx > 10000 || dy > 10000) return;
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    int dash_counter = 0;
    int dash_counter_modulus = (wh.x + wh.y) >> 7; // Naive average

    while (true) {
        if(!is_dashed || ((dash_counter / dash_counter_modulus) % 2 == 0)) {
            atomic_overlay_pixel(x1, y1, col, opacity, pixels, wh);
            for (int i = 1; i < thickness; i++) {
                atomic_overlay_pixel(x1 + i, y1, col, opacity, pixels, wh);
                atomic_overlay_pixel(x1 - i, y1, col, opacity, pixels, wh);
                atomic_overlay_pixel(x1, y1 + i, col, opacity, pixels, wh);
                atomic_overlay_pixel(x1, y1 - i, col, opacity, pixels, wh);
            }
        }
        dash_counter++;
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 <  dx) { err += dx; y1 += sy; }
    }
}

__device__ __forceinline__ void d_coordinate_to_pixel(
    const vec3& coordinate,
    bool &behind_camera,
    const quat& camera_direction,
    const vec3& camera_pos,
    const float fov,
    const float geom_mean_size,
    const Cuda::ivec2& wh,
    vec3& pixel)
{
    behind_camera = false;
    vec3 rotated = rotate_vector(coordinate - camera_pos, camera_direction);
    if (rotated.z <= 0) { behind_camera = true; return; }
    float scale = (geom_mean_size * fov) / rotated.z;
    pixel.x = scale * rotated.x + wh.x * 0.5f;
    pixel.y = -scale * rotated.y + wh.y * 0.5f;
    pixel.z = rotated.z;
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
