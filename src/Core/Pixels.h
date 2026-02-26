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
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

class Pixels{
public:
    int w;
    int h;
    vector<unsigned int> pixels;
    Pixels();
    Pixels(int width, int height);

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

    void scale_to_bounding_box(int box_w, int box_h, Pixels &scaled) const;

    void crop(int x, int y, int cw, int ch, Pixels &cropped) const;

    void crop_by_fractions(float crop_top, float crop_bottom, float crop_left, float crop_right, Pixels &cropped) const;

    int get_pixel_bilinear(double x, double y) const;

    void rotate_arbitrary_angle(double angle_rad, Pixels &rotated) const;

    bool is_empty() const;

    void add_border(int col, int thickness = 1);

    void overlay(Pixels p, int dx, int dy, double overlay_opacity_multiplier = 1);

    void overwrite(Pixels p, int dx, int dy);

    void fill_rect(int x, int y, int rw, int rh, int col);

    void fill_circle(double x, double y, double r, int col, double opa = 1);

    void fill_ring(double x, double y, double r_outer, double r_inner, int col, double opa = 1);

    void fill_ellipse(double x, double y, double rw, double rh, int col, double opa = 1);

    void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness);

    void rounded_rect(float x, float y, float rw, float rh, float r, int col);

    void flood_fill(int x, int y, int color);

    Pixels naive_scale_down(int scale_down_factor) const;

    void rotate_90_inverse(Pixels& rotated) const;
    void rotate_90(Pixels& rotated) const;

    void bicubic_scale(int new_width, int new_height, Pixels& result) const;
    float bicubic_weight(float t) const;

    void print_to_terminal();
};

// Free functions
Pixels create_alpha_from_intensities(const vector<vector<unsigned int>>& intensities);
Pixels create_pixels_from_2d_vector(const vector<vector<unsigned int>>& colors, int negative_intensity);
Pixels crop_by_alpha(const Pixels& p);
