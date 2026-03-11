#include <cuda_runtime.h>
#include "color.cuh"
#include "../Host_Device_Shared/vec.h"

namespace Cuda {

// Fill a circle on a pixel buffer
__device__ __forceinline__ void d_fill_circle(const Cuda::vec2& pixel, float r, int col, unsigned int* pixels, const Cuda::vec2& size, float opa=1.0f) {
    // breakout if outside of screen
    if (pixel.x + r < 0 || pixel.x - r >= size.x || pixel.y + r < 0 || pixel.y - r >= size.y)
        return;
    float r2 = r*r;
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            if (sdx + dy*dy < r2)
                d_atomic_overlay_pixel(pixel + vec2(dx, dy), col, opa, pixels, size);
        }
    }
}

__device__ __forceinline__ void bresenham(const Cuda::vec2& start, const Cuda::vec2& end, int col, float opacity, int thickness, unsigned int* pixels, const Cuda::vec2& screen_size) {
    int x1 = static_cast<int>(start.x), y1 = static_cast<int>(start.y);
    int x2 = static_cast<int>(end.x), y2 = static_cast<int>(end.y);
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    if (dx > 10000 || dy > 10000) return;
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        d_atomic_overlay_pixel(vec2(x1, y1), col, opacity, pixels, screen_size);
        for (int i = 1; i < thickness; i++) {
            d_atomic_overlay_pixel(vec2(x1 + i, y1), col, opacity, pixels, screen_size);
            d_atomic_overlay_pixel(vec2(x1 - i, y1), col, opacity, pixels, screen_size);
            d_atomic_overlay_pixel(vec2(x1, y1 + i), col, opacity, pixels, screen_size);
            d_atomic_overlay_pixel(vec2(x1, y1 - i), col, opacity, pixels, screen_size);
        }
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
    const Cuda::vec2& screen_size,
    vec3& out)
{
    behind_camera = false;
    vec3 rotated = rotate_vector(coordinate - camera_pos, camera_direction);
    if (rotated.z <= 0) { behind_camera = true; return; }
    float scale = (geom_mean_size * fov) / rotated.z;
    out.x = scale * rotated.x + screen_size.x * 0.5f;
    out.y = scale * rotated.y + screen_size.y * 0.5f;
    out.z = rotated.z;
}

}
