#pragma once

__device__ inline int geta(int color) { 
    return (color >> 24) & 0xFF; 
}

__device__ inline int colorlerp(int c1, int c2, float t) {
    int a1 = (c1 >> 24) & 0xFF; int r1 = (c1 >> 16) & 0xFF; int g1 = (c1 >> 8) & 0xFF; int b1 = c1 & 0xFF;
    int a2 = (c2 >> 24) & 0xFF; int r2 = (c2 >> 16) & 0xFF; int g2 = (c2 >> 8) & 0xFF; int b2 = c2 & 0xFF;
    int a = roundf((1 - t) * a1 + t * a2);
    int r = roundf((1 - t) * r1 + t * r2);
    int g = roundf((1 - t) * g1 + t * g2);
    int b = roundf((1 - t) * b1 + t * b2);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

__device__ __forceinline__ float square(float x) {
    return x * x;
}

__device__ int device_color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = geta(base_color) / 255.0f;
    float over_opacity = geta(over_color) / 255.0f * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = roundf(final_opacity * 255.0f);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

__device__ void set_pixel(int x, int y, int col, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    pixels[y * width + x] = col;
}

__device__ void overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int idx = y * width + x;
    int base = pixels[idx];
    int blended = device_color_combine(base, col, opacity);
    pixels[idx] = blended;
}
