#pragma once
#include <thrust/complex.h>
#include "../Host_Device_Shared/helpers.h"

__device__ __forceinline__ int d_argb(int a, int r, int g, int b){return (int(clamp(a,0,255))<<24)|
                                                                         (int(clamp(r,0,255))<<16)|
                                                                         (int(clamp(g,0,255))<<8 )|
                                                                         (int(clamp(b,0,255))    );}

__device__ __forceinline__ int d_geta(int color) { 
    return (color >> 24) & 0xFF; 
}
__device__ __forceinline__ int d_getr(int color) { 
    return (color >> 16) & 0xFF; 
}
__device__ __forceinline__ int d_getg(int color) { 
    return (color >> 8) & 0xFF; 
}
__device__ __forceinline__ int d_getb(int color) { 
    return color & 0xFF; 
}

__device__ __forceinline__ int d_colorlerp(int c1, int c2, float t) {
    int a1 = (c1 >> 24) & 0xFF; int r1 = (c1 >> 16) & 0xFF; int g1 = (c1 >> 8) & 0xFF; int b1 = c1 & 0xFF;
    int a2 = (c2 >> 24) & 0xFF; int r2 = (c2 >> 16) & 0xFF; int g2 = (c2 >> 8) & 0xFF; int b2 = c2 & 0xFF;
    int a = roundf((1 - t) * a1 + t * a2);
    int r = roundf((1 - t) * r1 + t * r2);
    int g = roundf((1 - t) * g1 + t * g2);
    int b = roundf((1 - t) * b1 + t * b2);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

__device__ __forceinline__ int d_color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = d_geta(base_color) / 255.0f;
    float over_opacity = d_geta(over_color) / 255.0f * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = roundf(final_opacity * 255.0f);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = d_colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

__device__ __forceinline__ void d_set_pixel(int x, int y, int col, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    pixels[y * width + x] = col;
}

__device__ __forceinline__ void d_overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    opacity = clamp(opacity, 0.0f, 1.0f);
    int idx = y * width + x;
    int base = pixels[idx];
    int blended = d_color_combine(base, col, opacity);
    pixels[idx] = blended;
}

__device__ __forceinline__ void d_naive_add_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    opacity = clamp(opacity, 0.0f, 1.0f);
    int idx = y * width + x;
    int base = pixels[idx];
    // Naively add each channel multiplied by opacity, clamping to 255
    int a = d_geta(base) + int(d_geta(col) * opacity);
    int r = d_getr(base) + int(d_getr(col) * opacity);
    int g = d_getg(base) + int(d_getg(col) * opacity);
    int b = d_getb(base) + int(d_getb(col) * opacity);
    int blended = d_argb(a, r, g, b);
    pixels[idx] = blended;
}

__device__ __forceinline__ void d_atomic_overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    opacity = clamp(opacity, 0.0f, 1.0f);
    int idx = y * width + x;

    unsigned int old_pixel = pixels[idx];
    int base = old_pixel;
    int blended = d_color_combine(base, col, opacity);
    int new_pixel = blended;
    atomicCAS(&pixels[idx], old_pixel, new_pixel);
}

__device__ __forceinline__ float d_linear_srgb_to_srgb(float x) {
    return x;
    if (x >= 0.0031308)
        return 1.055*pow(x, 1.0/2.4) - 0.055;
    return 12.92 * x;
}

__device__ __forceinline__ int d_OKLABtoRGB(int alpha, float L, float a, float b)
{
    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    return d_argb(
        alpha,
        256*d_linear_srgb_to_srgb(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s),
        256*d_linear_srgb_to_srgb(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s),
        256*d_linear_srgb_to_srgb(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s)
    );
}

__device__ __forceinline__ int d_complex_to_srgb(const thrust::complex<float>& c, float ab_dilation, float dot_radius) {
    float mag = abs(c);
    if(mag < 1e-7) return d_argb(255, 255, 255, 255); // white to dodge division by zero 
    thrust::complex<float> norm = (c * ab_dilation / mag + thrust::complex<float>(1,1)) * .5;
    float am = 2*atan(mag/dot_radius)/M_PI;
    return d_OKLABtoRGB(255, 1-.8*am, lerp(-.233888, .276216, norm.real()), lerp(-.311528, .198570, norm.imag()));
}

__device__ __forceinline__ int d_HSVtoRGB(float h, float s, float v, int alpha = 255) {
    float r_f, g_f, b_f;

    if (s == 0.0) {
        // Achromatic (grey)
        r_f = g_f = b_f = v;
    } else {
        h = fmod(double(h), 1.0) * 6.0;  // Hue sector [0, 6)
        int i = static_cast<int>(floor(h));
        float f = h - i;
        float p = v * (1.0 - s);
        float q = v * (1.0 - s * f);
        float t = v * (1.0 - s * (1.0 - f));

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
    return d_argb(alpha, r, g, b);
}
