// Everything in this file is inline, c-compatible, and therefore usable in both CUDA and C++.

#pragma once
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <cstdlib>
#include "math.h"
#include <cmath>
#include <sys/sysinfo.h>
#include <iostream>
#include <csignal>
#include "shared_precompiler_directives.h"

using namespace std;

HOST_DEVICE inline double sigmoid(double x){return 2/(1+exp(-x))-1;}
HOST_DEVICE inline double clamp(double val, double bottom, double top){return min(top, max(val, bottom));}
HOST_DEVICE inline double square(double x){return x * x;}
HOST_DEVICE inline double cube(double x){return x * x * x;}
HOST_DEVICE inline double fourth(double x){return square(square(x));}
HOST_DEVICE inline double smoother1(double x){return 3*x*x-2*x*x*x;} // We used to use this but not anymore
HOST_DEVICE inline double smoother2(double x){return x<.5 ? square(x)*2 : 1-square(1-x)*2;}
HOST_DEVICE inline double lerp(double a, double b, double w){return a*(1-w)+b*w;}
HOST_DEVICE inline float float_lerp(float a, float b, float w){return a*(1-w)+b*w;}
HOST_DEVICE inline glm::dvec3 veclerp(glm::dvec3 a, glm::dvec3 b, double w){return a*(1-w)+b*w;}
HOST_DEVICE inline glm::dvec4 veclerp(glm::dvec4 a, glm::dvec4 b, double w){return a*(1-w)+b*w;}
HOST_DEVICE inline double smoothlerp(double a, double b, double w){double v = smoother2(w);return a*(1-v)+b*v;}
HOST_DEVICE inline bool is_single_letter(const std::string& str) {return str.length() == 1 && isalpha(str[0]);}
HOST_DEVICE inline void print_vec3(glm::vec3 v){printf("vec3(%.3f, %.3f, %.3f)\n", v.x, v.y, v.z);}
HOST_DEVICE inline void print_vec4(glm::vec4 v){printf("vec4(%.3f, %.3f, %.3f, %.3f)\n", v.x, v.y, v.z, v.w);}
HOST_DEVICE inline double geom_mean(double x, double y) { return sqrt(x*y); }
HOST_DEVICE inline int signum(double x) { return (x > 0) - (x < 0); }

HOST_DEVICE inline double extended_mod(double a, double b) {
    double result = fmod(a, b);
    if (result < 0) {
        result += b;  // Ensures non-negative remainder
    }
    return result;
}

HOST_DEVICE inline glm::vec2 pixel_to_point(const glm::vec2& pixel, const glm::vec2& lx_ty, const glm::vec2& rx_by, const glm::vec2& wh) {
    const glm::vec2 flip(pixel.x, wh.y-1-pixel.y);
    return flip * (rx_by - lx_ty) / wh + lx_ty;
}

HOST_DEVICE inline glm::vec2 point_to_pixel(const glm::vec2& point, const glm::vec2& lx_ty, const glm::vec2& rx_by, const glm::vec2& wh) {
    const glm::vec2 flip = (point - lx_ty) * wh / (rx_by - lx_ty);
    return glm::vec2(flip.x, wh.y-1-flip.y);
}
