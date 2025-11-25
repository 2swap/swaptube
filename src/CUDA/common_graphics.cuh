#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "color.cuh"

// Fill a circle on a pixel buffer
__device__ __forceinline__ void d_fill_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    // breakout if outside of screen
    if (cx + r < 0 || cx - r >= width || cy + r < 0 || cy - r >= height)
        return;
    float r2 = r*r;
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            if (sdx + dy*dy < r2)
                d_atomic_overlay_pixel(cx + dx, cy + dy, col, opa, pixels, width, height);
        }
    }
}

__device__ __forceinline__ void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness, unsigned int* pixels, int width, int height) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    if (dx > 10000 || dy > 10000) return;
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        d_atomic_overlay_pixel(x1, y1, col, opacity, pixels, width, height);
        for (int i = 1; i < thickness; i++) {
            d_atomic_overlay_pixel(x1 + i, y1, col, opacity, pixels, width, height);
            d_atomic_overlay_pixel(x1 - i, y1, col, opacity, pixels, width, height);
            d_atomic_overlay_pixel(x1, y1 + i, col, opacity, pixels, width, height);
            d_atomic_overlay_pixel(x1, y1 - i, col, opacity, pixels, width, height);
        }
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 <  dx) { err += dx; y1 += sy; }
    }
}

__device__ __forceinline__ void d_coordinate_to_pixel(
    const glm::vec3& coordinate,
    bool &behind_camera,
    const glm::quat& camera_direction,
    const glm::vec3& camera_pos,
    const glm::quat& conjugate_camera_direction,
    float fov,
    float geom_mean_size,
    int width,
    int height,
    float& outx,
    float& outy,
    float& outz)
{
    behind_camera = false;
    glm::vec3 rotated = camera_direction * (coordinate - camera_pos) * conjugate_camera_direction;
    if (rotated.z <= 0) { behind_camera = true; return; }
    float scale = (geom_mean_size * fov) / rotated.z;
    outx = scale * rotated.x + width * 0.5f;
    outy = scale * rotated.y + height * 0.5f;
    outz = rotated.z;
}
