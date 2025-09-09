#pragma once
__device__ void device_fill_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    for (float dx = -r; dx < r; dx++) {
        float sdx = dx*dx;
        for (float dy = -r; dy < r; dy++) {
            if (sdx + dy*dy < r*r)
                device_atomic_overlay_pixel(cx + dx, cy + dy, col, opa, pixels, width, height);
        }
    }
}

