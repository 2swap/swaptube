#pragma once

#include <vector>
#include <stack>
#include <stdexcept>
#include <limits>
#include <iostream>
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/Color.h"

using namespace std;

inline constexpr int OPAQUE_BLACK = 0xFF000000;
inline constexpr int OPAQUE_WHITE = 0xFFFFFFFF;
inline constexpr int TRANSPARENT_BLACK = 0x00000000;
inline constexpr int TRANSPARENT_WHITE = 0x00FFFFFF;

class Pixels{
public:
    ivec2 wh;
    vector<unsigned int> pixels;
    Pixels();
    Pixels(const ivec2& dim);

    bool out_of_range(int x, int y) const;

    int get_pixel_carelessly(int x, int y) const;
    int get_pixel_carefully(int x, int y) const;

    void get_pixel_by_channels(int x, int y, int& a, int& r, int& g, int& b) const;

    int get_alpha(int x, int y) const;

    void set_pixel_carelessly(int x, int y, int col);

    void get_average_color(int x_start, int y_start, int x_end, int y_end,
                           int &avgA, int &avgR, int &avgG, int &avgB);

    void scale_to_bounding_box(const vec2& box, Pixels &scaled) const;

    void crop(const ivec2&, const ivec2&, Pixels &cropped) const;
    void crop_by_alpha(Pixels& cropped);

    void fill_rect(int x, int y, int rw, int rh, int col);

    Pixels naive_scale_down(int scale_down_factor) const;

    void bicubic_scale(const ivec2& new_wh, Pixels& result) const;

    void print_to_terminal();
};
