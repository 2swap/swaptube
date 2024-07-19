#pragma once

// Colors are everywhere. For the sake of speed, we do not give them a dedicated child class.
// They are ints under the hood, and are always 32-bit, 4-channel ARGB.

inline int argb_to_col(int a, int r, int g, int b){return (a<<24)+
                                                          (r<<16)+
                                                          (g<<8 )+
                                                          (b    );}
inline int geta(int col){return (col&0xff000000)>>24;}
inline int getr(int col){return (col&0x00ff0000)>>16;}
inline int getg(int col){return (col&0x0000ff00)>>8 ;}
inline int getb(int col){return (col&0x000000ff)    ;}
inline int coldist(int col1, int col2){return abs(geta(col1) - geta(col2)) +
                                              abs(getr(col1) - getr(col2)) +
                                              abs(getg(col1) - getg(col2)) +
                                              abs(getb(col1) - getb(col2));}
inline int rainbow(double x){return makecol(255,
                                            sin((x+1/3.)*M_PI*2)*128.+128,
                                            sin((x+2/3.)*M_PI*2)*128.+128,
                                            sin((x     )*M_PI*2)*128.+128);}
inline int colorlerp(int col1, int col2, double w){return makecol(round(lerp(geta(col1), geta(col2), w)),
                                                                  round(lerp(getr(col1), getr(col2), w)),
                                                                  round(lerp(getg(col1), getg(col2), w)),
                                                                  round(lerp(getb(col1), getb(col2), w)));}
inline string color_to_string(int c){return "(" + to_string(geta(c)) + ", " + to_string(getr(c)) + ", " + to_string(getg(c)) + ", " + to_string(getb(c)) + ")";}
inline void print_argb(int c){cout << color_to_string(c) << endl;}

int color_combine(int base_color, int over_color, double overlay_opacity_multiplier = 1){
    double base_opacity = geta(base_color)/255.;
    double over_opacity = geta(over_color)/255.*overlay_opacity_multiplier;
    double final_opacity = 1-(1-base_opacity)*(1-over_opacity);
    if(final_opacity == 0) return 0x00000000;
    int final_alpha = round(final_opacity*255.);
    double chroma_weight = over_opacity/final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight)&0x00ffffff;
    return (final_alpha<<24)|(final_rgb);
}

void hsv2rgb(double h, double s, double v, int& r, int& g, int& b)
{
    double      hh, p, q, t, ff;
    long        i;

    if(s <= 0.0) {       // < is bogus, just shuts up warnings
        r = v;
        g = v;
        b = v;
    }
    hh = h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = v * (1.0 - s);
    q = v * (1.0 - (s * ff));
    t = v * (1.0 - (s * (1.0 - ff)));

    switch(i) {
    case 0:
        r = v*255;
        g = t*255;
        b = p*255;
        break;
    case 1:
        r = q*255;
        g = v*255;
        b = p*255;
        break;
    case 2:
        r = p*255;
        g = v*255;
        b = t*255;
        break;
    case 3:
        r = p*255;
        g = q*255;
        b = v*255;
        break;
    case 4:
        r = t*255;
        g = p*255;
        b = v*255;
        break;
    case 5:
    default:
        r = v*255;
        g = p*255;
        b = q*255;
        break;
    }
}

bool inline_unit_tests_verbose = false;

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

// Unit test for makecol function
void makecol_ut() {
    int a = 0xde;
    int r = 0xad;
    int g = 0xbe;
    int b = 0xef;
    int result = makecol(a, r, g, b);
    int expected = 0xdeadbeef;

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "makecol_alpha_ut passed." << endl;
    } else {
        cout << "makecol_alpha_ut failed." << endl;
        exit(1);
    }
}

// Unit test for rainbow function
void rainbow_ut() {
    double x = 0.25;
    int result = rainbow(x);
    int expected = 4210688 + (255<<24); // Equivalent to makecol(255, 191, 64)

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
    int col1 = makecol(2, 255, 0, 0);  // Red
    int col2 = makecol(4, 0, 0, 255);  // Blue
    double w = 0.5;
    int result = colorlerp(col1, col2, w);
    int expected = makecol(3, 128, 0, 128);  // Purple

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
        int col1 = makecol(0, 198, 55, 18); // Transparent Random
        int col2 = makecol(4, 5, 6, 7);
        double w = 0.5;
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
        int col1 = makecol(134, 198, 55, 18); // Random Color
        int col2 = makecol(255, 5  , 6 , 7 ); // Random opaque color;
        double w = 0.5;
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
        int col1 = makecol(128, 0, 0, 128); // Semi-Opaque Blue
        int col2 = makecol(128, 128, 0, 0); // Semi-Opaque Red
        double w = 0.5;
        int result = color_combine(col1, col2);
        int expected = makecol(192, 85, 0, 43); // Opaquer Purple

        if (result == expected) {
            if(inline_unit_tests_verbose) cout << "color_combine_ut passed." << endl;
        } else {
            cout << "color_combine_ut step 3 failed." << endl;
            exit(1);
        }
    }
}

void run_inlines_unit_tests(){
    sigmoid_ut();
    min_ut();
    max_ut();
    square_ut();
    cube_ut();
    rgb_to_col_ut();
    smoother1_ut();
    smoother2_ut();
    coldist_ut();
    lerp_ut();
    makecol_ut();
    rainbow_ut();
    geta_ut();
    getr_ut();
    getg_ut();
    getb_ut();
    colorlerp_ut();
    latex_text_ut();
    color_combine_ut();
    cout << "inline tests passed." << endl;
}
