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

__device__ void atomic_overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int idx = y * width + x;

    unsigned int old_pixel = pixels[idx];
    int base = old_pixel;
    int blended = device_color_combine(base, col, opacity);
    int new_pixel = blended;
    atomicCAS(&pixels[idx], old_pixel, new_pixel);
}

__device__ int argb(int a, int r, int g, int b){return (a<<24)+
                                                       (r<<16)+
                                                       (g<<8 )+
                                                       (b    );}

__device__ int device_HSVtoRGB(double h, double s, double v, int alpha = 255) {
    double r_f, g_f, b_f;

    if (s == 0.0) {
        // Achromatic (grey)
        r_f = g_f = b_f = v;
    } else {
        h = fmod(h, 1.0) * 6.0;  // Hue sector [0, 6)
        int i = static_cast<int>(floor(h));
        double f = h - i;
        double p = v * (1.0 - s);
        double q = v * (1.0 - s * f);
        double t = v * (1.0 - s * (1.0 - f));

        switch (i) {
            case 0: r_f = v; g_f = t; b_f = p; break;
            case 1: r_f = q; g_f = v; b_f = p; break;
            case 2: r_f = p; g_f = v; b_f = t; break;
            case 3: r_f = p; g_f = q; b_f = v; break;
            case 4: r_f = t; g_f = p; b_f = v; break;
            case 5: default: r_f = v; g_f = p; b_f = q; break;
        }
    }

    // Scale to [0, 255] and clamp
    int r = clamp(static_cast<int>(round(r_f * 255.0)), 0, 255);
    int g = clamp(static_cast<int>(round(g_f * 255.0)), 0, 255);
    int b = clamp(static_cast<int>(round(b_f * 255.0)), 0, 255);
    return argb(alpha, r, g, b);
}
