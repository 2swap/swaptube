#pragma once

#include <string>
#include <cstdlib>
#include "math.h"
#include <cmath>

using namespace std;

inline double sigmoid(double x){return 2/(1+exp(-x))-1;}
inline double min(double a, double b){return a<b?a:b;}
inline double max(double a, double b){return a>b?a:b;}
inline double bound(double bottom, double val, double top){return min(top, max(val, bottom));}
inline double square(double x){return x * x;}
inline double cube(double x){return x * x * x;}
inline double fourth(double x){return square(square(x));}
inline double smoother1(double x){return 3*x*x-2*x*x*x;}
inline double smoother2(double x){return x<.5 ? square(x)*2 : 1-square(1-x)*2;}
inline double transparency_profile(double x){return x<.5 ? cube(x/.5) : 1;}
inline double fifo_curve(double x){return 1-fourth(2*x-1);} // fade in fade out
inline double fifo_curve_experimental(double x, double seconds_so_far, double seconds_left){return min(1, max(max(1-fourth(2*x-1), transparency_profile(seconds_so_far/2)), transparency_profile(seconds_left/2)));} // fade in fade out
inline double lerp(double a, double b, double w){return a*(1-w)+b*w;}
inline double smoothlerp(double a, double b, double w){double v = smoother2(w);return a*(1-v)+b*v;}
inline string latex_text(string in){return "\\text{" + in + "}";}
inline float fractional_part(float x) {return x - floor(x);}
inline string failout(string message){
    cerr << "======================================================\n";
    cerr << message << "\n";
    cerr << "======================================================\n";
    exit(EXIT_FAILURE);
}

bool inline_unit_tests_verbose = false;

// Unit test for sigmoid function
void sigmoid_ut() {
    double x = 1.0;
    double result = sigmoid(x);
    double expected = 0.462117157;
    double tolerance = 0.000001;

    if (abs(result - expected) < tolerance) {
        if(inline_unit_tests_verbose) cout << "sigmoid_ut passed." << endl;
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
        if(inline_unit_tests_verbose) cout << "min_ut passed." << endl;
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
        if(inline_unit_tests_verbose) cout << "max_ut passed." << endl;
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
        if(inline_unit_tests_verbose) cout << "square_ut passed." << endl;
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
        if(inline_unit_tests_verbose) cout << "cube_ut passed." << endl;
    } else {
        cout << "cube_ut failed." << endl;
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
        if(inline_unit_tests_verbose) cout << "smoother1_ut passed." << endl;
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
        if(inline_unit_tests_verbose) cout << "smoother2_ut passed." << endl;
    } else {
        cout << "smoother2_ut failed." << endl;
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
        if(inline_unit_tests_verbose) cout << "lerp_ut passed." << endl;
    } else {
        cout << "lerp_ut failed." << endl;
        exit(1);
    }
}

// Unit test for latex_text function
void latex_text_ut() {
    string input = "Hello";
    string result = latex_text(input);
    string expected = "\\text{Hello}";

    if (result == expected) {
        if(inline_unit_tests_verbose) cout << "latex_text_ut passed." << endl;
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
