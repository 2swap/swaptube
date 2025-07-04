#pragma once
#include <iomanip>
#include "inlines.h"

// Colors are everywhere. For the sake of speed, we do not give them a dedicated class.
// They are ints under the hood, and are always 32-bit, 4-channel ARGB.

inline int argb(int a, int r, int g, int b){return (a<<24)+
                                                   (r<<16)+
                                                   (g<<8 )+
                                                   (b    );}
inline int geta(int col){return (col&0xff000000)>>24;}
inline int getr(int col){return (col&0x00ff0000)>>16;}
inline int getg(int col){return (col&0x0000ff00)>>8 ;}
inline int getb(int col){return (col&0x000000ff)    ;}
inline int alpha_multiply(int col, float opacity){return (static_cast<int>(geta(col) * opacity) << 24) | col&0xffffff;}
inline int coldist(int col1, int col2){return abs(geta(col1) - geta(col2)) +
                                              abs(getr(col1) - getr(col2)) +
                                              abs(getg(col1) - getg(col2)) +
                                              abs(getb(col1) - getb(col2));}
inline int rainbow(float x){return argb(255,
                                            sin((x+1/3.)*M_PI*2)*128.+128,
                                            sin((x+2/3.)*M_PI*2)*128.+128,
                                            sin((x     )*M_PI*2)*128.+128);}
inline int colorlerp(int col1, int col2, float w){return argb(round(lerp(geta(col1), geta(col2), w)),
                                                                      round(lerp(getr(col1), getr(col2), w)),
                                                                      round(lerp(getg(col1), getg(col2), w)),
                                                                      round(lerp(getb(col1), getb(col2), w)));}
inline string color_to_string(int c){return "(" + to_string(geta(c)) + ", " + to_string(getr(c)) + ", " + to_string(getg(c)) + ", " + to_string(getb(c)) + ")";}
inline void print_argb(int c){cout << color_to_string(c) << endl;}

int color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = geta(base_color) / 255.0;
    float over_opacity = geta(over_color) / 255.0 * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = round(final_opacity * 255.0);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

int black_to_blue_to_white(double w){
    int rainbow_part1 = max(0.,min(1.,w*2-0))*255.;
    int rainbow_part2 = max(0.,min(1.,w*2-1))*255.;
    return argb(255, rainbow_part2, rainbow_part2, rainbow_part1);
}

int RGBtoOKLAB(int rgb)
{
    int r = getr(rgb);
    int g = getg(rgb);
    int b = getb(rgb);
    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
	float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
	float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

    float l_ = cbrtf(l);
    float m_ = cbrtf(m);
    float s_ = cbrtf(s);

    return argb(
        geta(rgb),
        0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_,
        1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_,
        0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_
    );
}

int OKLABtoRGB(int oklab)
{
    int L = getr(oklab);
    int a = getg(oklab);
    int b = getb(oklab);
    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    return argb(
        geta(oklab),
		+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
		-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
		-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s
    );
}

// Convert RGB to YUV
int RGBtoYUV(const int rgb) {
    int r = getr(rgb);
    int g = getg(rgb);
    int b = getb(rgb);

    // Conversion formulas
    double y_f = 0.299 * r + 0.587 * g + 0.114 * b;
    double u_f = -0.14713 * r - 0.28886 * g + 0.436 * b + 128; // Offset U and V by 128
    double v_f = 0.615 * r - 0.51499 * g - 0.10001 * b + 128;

    // Round and convert to integers
    int y = clamp(static_cast<int>(round(y_f)), 0, 255);
    int u = clamp(static_cast<int>(round(u_f)), 0, 255);
    int v = clamp(static_cast<int>(round(v_f)), 0, 255);
    return argb(geta(rgb), y, u, v);
}

// Convert YUV to RGB
int YUVtoRGB(const int yuv) {
    int y = getr(yuv);
    int u = getg(yuv)-128;
    int v = getb(yuv)-128;

    // Conversion formulas
    double r_f = y + 1.13983 * v;
    double g_f = y - 0.39465 * u - 0.58060 * v;
    double b_f = y + 2.03211 * u;

    // Round and convert to integers
    int r = clamp(static_cast<int>(round(r_f)), 0, 255);
    int g = clamp(static_cast<int>(round(g_f)), 0, 255);
    int b = clamp(static_cast<int>(round(b_f)), 0, 255);
    return argb(geta(yuv), r, g, b);
}

// Convert HSV to RGB
int HSVtoRGB(double h, double s, double v, int alpha = 255) {
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

int pendulum_color_old(double angle1, double angle2) {
    angle1 += M_PI;
    double y_f = 127.5 +  64 * sin(angle1 + angle2);
    double u_f = 127.5 + 128 * sin(angle1)*cos(angle2);
    double v_f = 127.5 + 128 * sin(angle2);

    int y = clamp(static_cast<int>(round(y_f)), 0, 255);
    int u = clamp(static_cast<int>(round(u_f)), 0, 255);
    int v = clamp(static_cast<int>(round(v_f)), 0, 255);
    return YUVtoRGB(argb(255, y, u, v));
}

int pendulum_color(double angle1, double angle2, double p1, double p2) {
    float sa1 = sin(angle1) + 0.000001;
    float sa2 = sin(angle2);
    float h = atan2(sa2, sa1)/6.283+1;
    float s = min((square(sa1) + square(sa2))*5.,1.);
    float v = 1-min(.1 * sqrt(p1*p1+p2*p2), 1.0);
    return HSVtoRGB(h, s, v);
}

string latex_color(unsigned int color, string text) {
    // Mask out the alpha channel
    unsigned int rgb = color & 0x00FFFFFF;

    // Convert to a hex string
    stringstream ss;
    ss << "\\textcolor{#" << hex << setw(6) << setfill('0') << rgb << "}{" << text << "}";

    return ss.str();
}

// Unit test for coldist function
void coldist_ut() {
    int col1 = 0xF0A0B0C0;
    int col2 = 0xF1A2B3C4;
    int result = coldist(col1, col2);
    int expected = 10;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "coldist_ut passed." << endl;
    } else {
        cout << "coldist_ut failed." << endl;
        exit(1);
    }
}

// Unit test for argb function
void argb_ut() {
    int a = 0xde;
    int r = 0xad;
    int g = 0xbe;
    int b = 0xef;
    int result = argb(a, r, g, b);
    int expected = 0xdeadbeef;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "argb_alpha_ut passed." << endl;
    } else {
        cout << "argb_alpha_ut failed." << endl;
        exit(1);
    }
}

// Unit test for rainbow function
void rainbow_ut() {
    float x = 0.25;
    int result = rainbow(x);
    int expected = 4210688 + (255<<24); // Equivalent to argb(255, 191, 64)

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "rainbow_ut passed." << endl;
    } else {
        cout << "rainbow_ut failed." << endl;
        exit(1);
    }
}

// Unit test for geta function
void geta_ut() {
    int col = 0xFFAABBCC;
    int result = geta(col);
    int expected = 255;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "geta_ut passed." << endl;
    } else {
        cout << "geta_ut failed." << endl;
        exit(1);
    }
}

// Unit test for getr function
void getr_ut() {
    int col = 0xFFAABBCC;
    int result = getr(col);
    int expected = 170;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "getr_ut passed." << endl;
    } else {
        cout << "getr_ut failed." << endl;
        exit(1);
    }
}

// Unit test for getg function
void getg_ut() {
    int col = 0xFFAABBCC;
    int result = getg(col);
    int expected = 187;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "getg_ut passed." << endl;
    } else {
        cout << "getg_ut failed." << endl;
        exit(1);
    }
}

// Unit test for getb function
void getb_ut() {
    int col = 0xFFAABBCC;
    int result = getb(col);
    int expected = 204;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "getb_ut passed." << endl;
    } else {
        cout << "getb_ut failed." << endl;
        exit(1);
    }
}

// Unit test for colorlerp function
void colorlerp_ut() {
    int col1 = argb(2, 255, 0, 0);  // Red
    int col2 = argb(4, 0, 0, 255);  // Blue
    float w = 0.5;
    int result = colorlerp(col1, col2, w);
    int expected = argb(3, 128, 0, 128);  // Purple

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "colorlerp_ut passed." << endl;
    } else {
        cout << "colorlerp_ut failed." << endl;
        exit(1);
    }
}

// Unit test for color_combine function
void color_combine_ut() {
    {
        int col1 = argb(0, 198, 55, 18); // Transparent Random
        int col2 = argb(4, 5, 6, 7);
        int result = color_combine(col1, col2);
        int expected = col2;

        if (result == expected) {
            if(inline_unit_tests_verbose) cout << "color_combine_ut passed." << endl;
        } else {
            cout << "color_combine_ut step 1 failed.";
            exit(1);
        }
    }

    {
        int col1 = argb(134, 198, 55, 18); // Random Color
        int col2 = argb(255, 5  , 6 , 7 ); // Random opaque color;
        int result = color_combine(col1, col2);
        int expected = col2;

        if (result == expected) {
            if(inline_unit_tests_verbose) cout << "color_combine_ut passed." << endl;
        } else {
            cout << "color_combine_ut step 2 failed." << endl;
            exit(1);
        }
    }

    {
        int col1 = argb(128, 0, 0, 128); // Semi-Opaque Blue
        int col2 = argb(128, 128, 0, 0); // Semi-Opaque Red
        int result = color_combine(col1, col2);
        int expected = argb(192, 85, 0, 43); // Opaquer Purple

        if (result == expected) {
            if(inline_unit_tests_verbose) cout << "color_combine_ut passed." << endl;
        } else {
            cout << "color_combine_ut step 3 failed." << endl;
            exit(1);
        }
    }
}

void run_color_unit_tests(){
    coldist_ut();
    argb_ut();
    rainbow_ut();
    geta_ut();
    getr_ut();
    getg_ut();
    getb_ut();
    colorlerp_ut();
    color_combine_ut();
}
