#pragma once

#include <vector>
#include <stack>
#include <stdexcept>
#include <limits>
#include <iostream>
#include "../Host_Device_Shared/helpers.h"
#include "Color.h"

using namespace std;

inline constexpr int OPAQUE_BLACK = 0xFF000000;
inline constexpr int OPAQUE_WHITE = 0xFFFFFFFF;
inline constexpr int TRANSPARENT_BLACK = 0x00000000;
inline constexpr int TRANSPARENT_WHITE = 0x00FFFFFF;

extern "C" void cuda_overlay(
    unsigned int* h_background, const vec2& b_size,
    unsigned int* h_foreground, const vec2& f_size,
    const vec2& center, const float opacity, const float angle_rad);

class Pixels{
public:
    vec2 size;
    vector<unsigned int> pixels;
    Pixels();
    Pixels(const vec2& dim);

    bool out_of_range(int x, int y) const;

    int get_pixel_carelessly(int x, int y) const;
    int get_pixel_carefully(int x, int y) const;

    void overlay_pixel(int x, int y, int col, double overlay_opacity_multiplier = 1);

    void get_pixel_by_channels(int x, int y, int& a, int& r, int& g, int& b) const;

    int get_alpha(int x, int y) const;

    void set_pixel_carelessly(int x, int y, int col);
    void set_pixel_carefully(int x, int y, int col);

    void darken(float factor);

    void set_alpha(int x, int y, int a);

    void get_average_color(int x_start, int y_start, int x_end, int y_end,
                           int &avgA, int &avgR, int &avgG, int &avgB);

    void scale_to_bounding_box(const vec2& box_dim, Pixels &scaled) const;

    void crop(const vec2& pos, const vec2& dimensions, Pixels &cropped) const;

    void crop_by_fractions(const vec2& crop_top_left, const vec2& crop_bottom_right, Pixels &cropped) const;

    int get_pixel_bilinear(double x, double y) const;

    void add_border(int col, int thickness = 1);

    void overlay(const Pixels& p, const vec2& pos, double overlay_opacity_multiplier = 1);

    void overwrite(const Pixels& p, const vec2& pos);

    void fill_rect(const vec2& pos, const vec2& dimensions, int col);

    void fill_circle(const vec2& pos, double r, int col, double opa = 1);

    void fill_ring(const vec2& pos, double r_outer, double r_inner, int col, double opa = 1);

    void fill_ellipse(const vec2& pos, const vec2& dimensions, int col, double opa = 1);

    void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness);

    void rounded_rect(const vec2& pos, const vec2& dimensions, float r, int col);

    Pixels naive_scale_down(int scale_down_factor) const;

    void bicubic_scale(const vec2& new_dim, Pixels& result) const;
    float bicubic_weight(float t) const;

    void print_to_terminal();
};

// Free functions
Pixels create_alpha_from_intensities(const vector<vector<unsigned int>>& intensities);
Pixels crop_by_alpha(const Pixels& p);
