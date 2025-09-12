#pragma once

__device__ inline int device_geta(int color) { 
    return (color >> 24) & 0xFF; 
}
__device__ inline int device_getr(int color) { 
    return (color >> 16) & 0xFF; 
}
__device__ inline int device_getg(int color) { 
    return (color >> 8) & 0xFF; 
}
__device__ inline int device_getb(int color) { 
    return color & 0xFF; 
}

__device__ inline int device_colorlerp(int c1, int c2, float t) {
    int a1 = (c1 >> 24) & 0xFF; int r1 = (c1 >> 16) & 0xFF; int g1 = (c1 >> 8) & 0xFF; int b1 = c1 & 0xFF;
    int a2 = (c2 >> 24) & 0xFF; int r2 = (c2 >> 16) & 0xFF; int g2 = (c2 >> 8) & 0xFF; int b2 = c2 & 0xFF;
    int a = roundf((1 - t) * a1 + t * a2);
    int r = roundf((1 - t) * r1 + t * r2);
    int g = roundf((1 - t) * g1 + t * g2);
    int b = roundf((1 - t) * b1 + t * b2);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

__device__ int device_color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = device_geta(base_color) / 255.0f;
    float over_opacity = device_geta(over_color) / 255.0f * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = roundf(final_opacity * 255.0f);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = device_colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

__device__ void device_set_pixel(int x, int y, int col, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    pixels[y * width + x] = col;
}

__device__ void device_overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    opacity = clamp(opacity, 0.0f, 1.0f);
    int idx = y * width + x;
    int base = pixels[idx];
    int blended = device_color_combine(base, col, opacity);
    pixels[idx] = blended;
}

__device__ void device_atomic_overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    opacity = clamp(opacity, 0.0f, 1.0f);
    int idx = y * width + x;

    unsigned int old_pixel = pixels[idx];
    int base = old_pixel;
    int blended = device_color_combine(base, col, opacity);
    int new_pixel = blended;
    atomicCAS(&pixels[idx], old_pixel, new_pixel);
}

__device__ int device_argb(int a, int r, int g, int b){return (int(clamp(0,a,255))<<24)|
                                                              (int(clamp(0,r,255))<<16)|
                                                              (int(clamp(0,g,255))<<8 )|
                                                              (int(clamp(0,b,255))    );}

__device__ float device_linear_srgb_to_srgb(float x) {
    return x;
	if (x >= 0.0031308)
		return 1.055*pow(x, 1.0/2.4) - 0.055;
	return 12.92 * x;
}

__device__ int device_OKLABtoRGB(int alpha, float L, float a, float b)
{
    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    return device_argb(
        alpha,
		256*device_linear_srgb_to_srgb(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s),
		256*device_linear_srgb_to_srgb(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s),
		256*device_linear_srgb_to_srgb(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s)
    );
}

__device__ int device_HSVtoRGB(float h, float s, float v, int alpha = 255) {
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
    return device_argb(alpha, r, g, b);
}
