#pragma once

#include <iomanip>
#include <cstdint>
#include "../Host_Device_Shared/helpers.h"

// Colors are everywhere. For the sake of speed, we do not give them a dedicated class.
// They are ints under the hood, and are always 32-bit, 4-channel ARGB.

SHARED_FILE_PREFIX

HOST_DEVICE inline uint32_t argb(int a, int r, int g, int b){return (a<<24)|
                                                   (r<<16)|
                                                   (g<<8 )|
                                                   (b    );}
HOST_DEVICE inline int geta(int col){return (col&0xff000000)>>24;}
HOST_DEVICE inline int getr(int col){return (col&0x00ff0000)>>16;}
HOST_DEVICE inline int getg(int col){return (col&0x0000ff00)>>8 ;}
HOST_DEVICE inline int getb(int col){return (col&0x000000ff)    ;}

HOST_DEVICE inline uint32_t rainbow(float x){return argb(255,
                                            sin((x+1/3.)*M_PI*2)*128.+128,
                                            sin((x+2/3.)*M_PI*2)*128.+128,
                                            sin((x     )*M_PI*2)*128.+128);}

HOST_DEVICE inline uint32_t colorlerp(int col1, int col2, float w){return argb(round(lerp(geta(col1), geta(col2), w)),
                                                                      round(lerp(getr(col1), getr(col2), w)),
                                                                      round(lerp(getg(col1), getg(col2), w)),
                                                                      round(lerp(getb(col1), getb(col2), w)));}

HOST_DEVICE inline uint32_t color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = geta(base_color) / 255.0;
    float over_opacity = geta(over_color) / 255.0 * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = round(final_opacity * 255.0);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

HOST_DEVICE inline uint32_t black_to_blue_to_white(double w){
    int rainbow_part1 = max(0.,min(1.,w*2-0))*255.;
    int rainbow_part2 = max(0.,min(1.,w*2-1))*255.;
    return argb(255, rainbow_part2, rainbow_part2, rainbow_part1);
}

// Convert HSV to RGB
// h, s, v are in the range [0, 1]
HOST_DEVICE inline uint32_t HSVtoRGB(double h, double s, double v, int alpha = 255) {
    double r_f, g_f, b_f;

    if (s == 0.0) {
        // Achromatic (grey)
        r_f = g_f = b_f = v;
    } else {
        h = fmod(h, 1.0) * 6.0;  // Hue sector [0, 6)
        int i = h;
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

HOST_DEVICE inline uint32_t pendulum_color(double angle1, double angle2, double p1, double p2) {
    float sa1 = sin(angle1) + 0.000001;
    float sa2 = sin(angle2);
    float h = atan2(sa2, sa1)/6.283+1;
    float s = min((square(sa1) + square(sa2))*5.,1.);
    float v = 1-min(.1 * sqrt(p1*p1+p2*p2), 1.0);
    return HSVtoRGB(h, s, v);
}

HOST_DEVICE inline float linear_srgb_to_srgb(float x) {
    return x;
    if (x >= 0.0031308)
        return 1.055*pow(x, 1.0/2.4) - 0.055;
    return 12.92 * x;
}

HOST_DEVICE inline uint32_t OKLABtoRGB(int alpha, float L, float a, float b)
{
    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    return argb(
        clamp(alpha, 0, 255),
        clamp(static_cast<int>(round(255*linear_srgb_to_srgb(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s))), 0, 255),
        clamp(static_cast<int>(round(255*linear_srgb_to_srgb(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s))), 0, 255),
        clamp(static_cast<int>(round(255*linear_srgb_to_srgb(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s))), 0, 255)
    );
}

SHARED_FILE_SUFFIX
