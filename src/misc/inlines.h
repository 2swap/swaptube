#pragma once

#include <glm/glm.hpp>
#include <string>
#include <cstdlib>
#include "math.h"
#include <cmath>
#include <sys/sysinfo.h>
#include <iostream>
#include <csignal>

using namespace std;

inline double sigmoid(double x){return 2/(1+exp(-x))-1;}
inline double bound(double bottom, double val, double top){return min(top, max(val, bottom));}
inline double square(double x){return x * x;}
inline double cube(double x){return x * x * x;}
inline double fourth(double x){return square(square(x));}
inline double smoother1(double x){return 3*x*x-2*x*x*x;}
inline double smoother2(double x){return x<.5 ? square(x)*2 : 1-square(1-x)*2;}
inline double lerp(double a, double b, double w){return a*(1-w)+b*w;}
inline glm::dvec3 veclerp(glm::dvec3 a, glm::dvec3 b, double w){return a*(1-w)+b*w;}
inline glm::dvec4 veclerp(glm::dvec4 a, glm::dvec4 b, double w){return a*(1-w)+b*w;}
inline double smoothlerp(double a, double b, double w){double v = smoother2(w);return a*(1-v)+b*v;}
inline string latex_text(string in){return "\\text{" + in + "}";}
inline bool is_single_letter(const std::string& str) {return str.length() == 1 && isalpha(str[0]);}

double extended_mod(double a, double b) {
    double result = fmod(a, b);
    if (result < 0) {
        result += b;  // Ensures non-negative remainder
    }
    return result;
}

void signal_handler(int signal) {
    if (signal == SIGINT) {
        throw runtime_error("Control-C interrupt detected. Exiting gracefully.");
    }
}

string replace_substring(string str, const string& from, const string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Move past the replacement to avoid infinite loop
    }
    return str;
}

long get_free_memory() {
    struct sysinfo memInfo;

     if (sysinfo(&memInfo) != 0) {
         perror("sysinfo");
         throw runtime_error("Unable to call sysinfo to determine system memory");
     }

     // Free memory in mb
     double free_memory = static_cast<double>(memInfo.freeram) * memInfo.mem_unit / square(1024);

     return free_memory;
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
    square_ut();
    cube_ut();
    smoother1_ut();
    smoother2_ut();
    lerp_ut();
    latex_text_ut();
}
