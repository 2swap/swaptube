
// File: Convolution.hpp
#pragma once

#include <stack>
#include <cassert>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "../IO/VisualMedia.h"

struct StepResult {
    int max_x;
    int max_y;
    Pixels map;
    Pixels induced1;
    Pixels induced2;
    Pixels current_p1;
    Pixels current_p2;
    Pixels intersection;

    StepResult(int mx, int my, Pixels cm, Pixels i1, Pixels i2, Pixels p1, Pixels p2, Pixels i)
            : max_x(mx), max_y(my), map(cm), induced1(i1), induced2(i2), current_p1(p1), current_p2(p2), intersection(i) {}
};

class TranslatedPixels {
public:
    int translation_x;
    int translation_y;
    Pixels pixels;

    TranslatedPixels(int width, int height, int tx, int ty)
        : translation_x(tx), translation_y(ty), pixels(width, height) {}

    TranslatedPixels(const Pixels& p, int tx, int ty)
        : translation_x(tx), translation_y(ty), pixels(p) {}

    TranslatedPixels(const TranslatedPixels& p, int tx, int ty)
        : translation_x(tx+p.translation_x), translation_y(ty+p.translation_y), pixels(p.pixels) {}

    inline bool out_of_range(int x, int y) const {
        return pixels.out_of_range(x - translation_x, y - translation_y);
    }

    inline int get_pixel(int x, int y) const {
        if (out_of_range(x, y)) return 0;
        return pixels.get_pixel_carelessly(x - translation_x, y - translation_y);
    }

    inline void set_pixel(int x, int y, int col) {
        if (out_of_range(x, y)) return;
        pixels.set_pixel_carelessly(x - translation_x, y - translation_y, col);
    }

    inline int get_alpha(int x, int y) const {
        return geta(get_pixel(x, y));
    }

    inline bool is_empty() const {
        return pixels.is_empty();
    }
};

Pixels convolve_map(const Pixels& p1, const Pixels& p2, int& max_x, int& max_y);

void flood_fill(Pixels& ret, const Pixels& p, int start_x, int start_y, int color);

Pixels segment(const Pixels& p, unsigned int& id);

void flood_fill_connected_to_opaque(const Pixels& p, Pixels& connected_to_opaque, int x, int y);

Pixels remove_unconnected_components(const Pixels& p);

TranslatedPixels intersect(const TranslatedPixels& tp1, const TranslatedPixels& tp2);

TranslatedPixels unify(const TranslatedPixels& tp1, const TranslatedPixels& tp2);

TranslatedPixels subtract(const TranslatedPixels& original, const TranslatedPixels& to_subtract);

void flood_fill_copy_shape(const TranslatedPixels& source, TranslatedPixels& destination, int start_x, int start_y);

TranslatedPixels induce(const TranslatedPixels& original, const TranslatedPixels& intersection);

Pixels colorize_segments(const Pixels& segmented);

int count_pixels_with_color(const Pixels& p, const unsigned int color);

TranslatedPixels erase_low_iou(const TranslatedPixels& intersection, const TranslatedPixels& unified, float threshold);

vector<StepResult> find_intersections(const Pixels& p1, const Pixels& p2);
