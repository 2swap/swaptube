#pragma once

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

// Make a circular gradient on a pixel buffer
__device__ __forceinline__ void d_gradient_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    // breakout if outside of screen
    if (cx + r < 0 || cx - r >= width || cy + r < 0 || cy - r >= height)
        return;
    float r2 = r*r;
    for (int x = cx-r; x < cx+r; x++) {
        float sdx = (x - cx)*(x - cx);
        for (int y = cy-r; y < cy+r; y++) {
            float sdy = (y - cy)*(y - cy);
            float dist2 = (sdx + sdy) / r2;
            if (dist2 < 1.0f) {
                //float final_opa = opa / (1 + 2400 /** dist2*/ * dist2);
                float final_opa = opa / (1 + 2400 * dist2 * dist2);
                final_opa = fminf(final_opa, 1.0f);
                //final_opa /= 2;
                d_atomic_overlay_pixel(x, y, col, final_opa, pixels, width, height);
            }
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

