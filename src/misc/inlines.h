#pragma once

#include <string>
#include <cstdlib>
#include "math.h"
#include <cmath>

using namespace std;

inline double sigmoid(double x){return 2/(1+exp(-x))-1;}
inline double min(double a, double b){return a<b?a:b;}
inline double max(double a, double b){return a>b?a:b;}
inline double square(double x){return x * x;}
inline double cube(double x){return x * x * x;}
inline double fourth(double x){return square(square(x));}
inline int rgb_to_col(int r, int g, int b){return r<<16|g<<8|b;}
inline double smoother1(double x){return 3*x*x-2*x*x*x;}
inline double smoother2(double x){return x<.5 ? square(x)*2 : 1-square(1-x)*2;}
inline double smoother3(double x){return x<.5 ? cube(x)*4 : 1-cube(1-x)*4;}
inline double transparency_profile(double x){return x<.6 ? cube(x/.6) : 1;}
inline double fifo_curve(double x){return 1-fourth(2*x-1);} // fade in fade out
inline double fifo_curve_experimental(double x, double seconds_so_far, double seconds_left){return min(1, max(max(1-fourth(2*x-1), transparency_profile(seconds_so_far/2)), transparency_profile(seconds_left/2)));} // fade in fade out
inline double lerp(double a, double b, double w){return a*(1-w)+b*w;}
inline int makecol(int r, int g, int b){return (255<<24)+(r<<16)+(g<<8)+b;}
inline int makecol(int a, int r, int g, int b){return (a<<24)+(r<<16)+(g<<8)+b;}
inline int geta(int col){return (col&0xff000000)>>24;}
inline int getr(int col){return (col&0x00ff0000)>>16;}
inline int getg(int col){return (col&0x0000ff00)>>8;}
inline int getb(int col){return (col&0x000000ff);}
inline int coldist(int col1, int col2){return abs(geta(col1) - geta(col2)) + abs(getr(col1) - getr(col2)) + abs(getg(col1) - getg(col2)) + abs(getb(col1) - getb(col2));}
inline int rainbow(double x){return makecol(sin((x+1/3.)*M_PI*2)*128.+128, sin((x+2/3.)*M_PI*2)*128.+128, sin(x*M_PI*2)*128.+128);}
inline int colorlerp(int col1, int col2, double w){return makecol(lerp(getr(col1), getr(col2), w), lerp(getg(col1), getg(col2), w), lerp(getb(col1), getb(col2), w));}
inline string latex_text(string in){return "\\text{" + in + "}";}

// Unit test for sigmoid function
void sigmoid_ut() {
    double x = 1.0;
    double result = sigmoid(x);
    double expected = 0.462117157;
    double tolerance = 0.000001;

    if (abs(result - expected) < tolerance) {
        cout << "sigmoid_ut passed." << endl;
    } else {
        cout << "sigmoid_ut failed." << endl;
        exit(1);
    }
}

// Unit test for min function
void min_ut() {
    double a = 3.0;
    double b = 2.0;
    double result = min(a, b);
    double expected = 2.0;

    if (result == expected) {
        cout << "min_ut passed." << endl;
    } else {
        cout << "min_ut failed." << endl;
        exit(1);
    }
}

// Unit test for max function
void max_ut() {
    double a = 3.0;
    double b = 2.0;
    double result = max(a, b);
    double expected = 3.0;

    if (result == expected) {
        cout << "max_ut passed." << endl;
    } else {
        cout << "max_ut failed." << endl;
        exit(1);
    }
}

// Unit test for square function
void square_ut() {
    double x = 2.5;
    double result = square(x);
    double expected = 6.25;

    if (result == expected) {
        cout << "square_ut passed." << endl;
    } else {
        cout << "square_ut failed." << endl;
        exit(1);
    }
}

// Unit test for cube function
void cube_ut() {
    double x = 2.0;
    double result = cube(x);
    double expected = 8.0;

    if (result == expected) {
        cout << "cube_ut passed." << endl;
    } else {
        cout << "cube_ut failed." << endl;
        exit(1);
    }
}

// Unit test for rgb_to_col function
void rgb_to_col_ut() {
    int r = 255;
    int g = 128;
    int b = 64;
    int result = rgb_to_col(r, g, b);
    int expected = 16744512;

    if (result == expected) {
        cout << "rgb_to_col_ut passed." << endl;
    } else {
        cout << "rgb_to_col_ut failed." << endl;
        exit(1);
    }
}

// Unit test for smoother1 function
void smoother1_ut() {
    double x = 0.75;
    double result = smoother1(x);
    double expected = 0.84375;

    double tolerance = 0.000001;
    if (abs(result - expected) < tolerance) {
        cout << "smoother1_ut passed." << endl;
    } else {
        cout << "smoother1_ut failed." << endl;
        exit(1);
    }
}

// Unit test for smoother2 function
void smoother2_ut() {
    double x = 0.2;
    double result = smoother2(x);
    double expected = 0.08;

    double tolerance = 0.000001;
    if (abs(result - expected) < tolerance) {
        cout << "smoother2_ut passed." << endl;
    } else {
        cout << "smoother2_ut failed." << endl;
        exit(1);
    }
}

// Unit test for smoother3 function
void smoother3_ut() {
    double x = 0.2;
    double result = smoother3(x);
    double expected = 0.032;

    double tolerance = 0.000001;
    if (abs(result - expected) < tolerance) {
        cout << "smoother3_ut passed." << endl;
    } else {
        cout << "smoother3_ut failed." << endl;
        exit(1);
    }
}

// Unit test for coldist function
void coldist_ut() {
    int col1 = 0xF0A0B0C0;
    int col2 = 0xF1A2B3C4;
    int result = coldist(col1, col2);
    int expected = 10;

    if (result == expected) {
        cout << "coldist_ut passed." << endl;
    } else {
        cout << "coldist_ut failed." << endl;
        exit(1);
    }
}

// Unit test for lerp function
void lerp_ut() {
    double a = 10.0;
    double b = 20.0;
    double w = 0.25;
    double result = lerp(a, b, w);
    double expected = 12.5;

    double tolerance = 0.000001;
    if (abs(result - expected) < tolerance) {
        cout << "lerp_ut passed." << endl;
    } else {
        cout << "lerp_ut failed." << endl;
        exit(1);
    }
}

// Unit test for makecol function
void makecol_ut() {
    int r = 255;
    int g = 128;
    int b = 64;
    int result = makecol(r, g, b);
    int expected = 16744512 + (255<<24);

    if (result == expected) {
        cout << "makecol_ut passed." << endl;
    } else {
        cout << "makecol_ut failed." << endl;
        exit(1);
    }
}

// Unit test for makecol function with alpha
void makecol_alpha_ut() {
    int a = 10;
    int r = 255;
    int g = 128;
    int b = 64;
    int result = makecol(a, r, g, b);
    int expected = 184516672;

    if (result == expected) {
        cout << "makecol_alpha_ut passed." << endl;
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
        cout << "rainbow_ut passed." << endl;
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
        cout << "geta_ut passed." << endl;
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
        cout << "getr_ut passed." << endl;
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
        cout << "getg_ut passed." << endl;
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
        cout << "getb_ut passed." << endl;
    } else {
        cout << "getb_ut failed." << endl;
        exit(1);
    }
}

// Unit test for colorlerp function
void colorlerp_ut() {
    int col1 = makecol(255, 0, 0);  // Red
    int col2 = makecol(0, 0, 255);  // Blue
    double w = 0.5;
    int result = colorlerp(col1, col2, w);
    int expected = makecol(127, 0, 127);  // Purple

    if (result == expected) {
        cout << "colorlerp_ut passed." << endl;
    } else {
        cout << "colorlerp_ut failed." << endl;
        exit(1);
    }
}

// Unit test for latex_text function
void latex_text_ut() {
    string input = "Hello";
    string result = latex_text(input);
    string expected = "\\text{Hello}";

    if (result == expected) {
        cout << "latex_text_ut passed." << endl;
    } else {
        cout << "latex_text_ut failed." << endl;
        exit(1);
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
    smoother3_ut();
    coldist_ut();
    lerp_ut();
    makecol_ut();
    makecol_alpha_ut();
    rainbow_ut();
    geta_ut();
    getr_ut();
    getg_ut();
    getb_ut();
    colorlerp_ut();
    latex_text_ut();
}