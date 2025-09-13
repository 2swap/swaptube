#pragma once

// Fill a circle on a pixel buffer
__device__ void device_fill_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    // breakout if outside of screen
    if (cx + r < 0 || cx - r >= width || cy + r < 0 || cy - r >= height)
        return;
    float r2 = r*r;
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            if (sdx + dy*dy < r2)
                device_atomic_overlay_pixel(cx + dx, cy + dy, col, opa, pixels, width, height);
        }
    }
}

// Make a circular gradient on a pixel buffer
__device__ void device_gradient_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    // breakout if outside of screen
    if (cx + r < 0 || cx - r >= width || cy + r < 0 || cy - r >= height)
        return;
    float r2 = r*r;
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            float dist2 = (sdx + dy*dy) / r2;
            if (dist2 < 1.0f) {
                //float final_opa = opa / (1 + 2400 /** dist2*/ * dist2);
                float final_opa = opa / (1 + 2400 * dist2 * dist2);
                //final_opa /= 2;
                device_atomic_overlay_pixel(cx + dx, cy + dy, col, final_opa, pixels, width, height);
            }
        }
    }
}
