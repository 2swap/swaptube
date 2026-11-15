#include "Pixels.h"

#include <unistd.h>
#include <sys/ioctl.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../IO/Writer.h"

extern "C" int cuda_bicubic_scale(const unsigned int* input_pixels, const ivec2& input_wh, unsigned int* output_pixels, const ivec2& output_wh);

Pixels::Pixels() : wh(0,0), pixels(0) {}
Pixels::Pixels(const ivec2& dim) : wh(dim), pixels(dim.x*dim.y) {}

bool Pixels::out_of_range(int x, int y) const {
    return x < 0 || x >= wh.x || y < 0 || y >= wh.y;
}

int Pixels::get_pixel_carelessly(int x, int y) const {
    return pixels[wh.x*y+x];
}

int Pixels::get_pixel_carefully(int x, int y) const {
    if(out_of_range(x, y)) return TRANSPARENT_BLACK; // 0
    return pixels[wh.x*y+x];
}

void Pixels::get_pixel_by_channels(int x, int y, int& a, int& r, int& g, int& b) const {
    int col = get_pixel_carefully(x,y);
    a=geta(col);
    r=getr(col);
    g=getg(col);
    b=getb(col);
}

int Pixels::get_alpha(int x, int y) const {
    return geta(get_pixel_carefully(x,y));
}

void Pixels::set_pixel_carelessly(int x, int y, int col) {
    pixels[wh.x*y+x] = col;
}

void Pixels::get_average_color(int x_start, int y_start, int x_end, int y_end,
                           int &avgA, int &avgR, int &avgG, int &avgB) {
    long long sumA = 0, sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    
    // Loop over the rectangular region.
    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            int a_pixel, r_pixel, g_pixel, b_pixel;
            get_pixel_by_channels(x, y, a_pixel, r_pixel, g_pixel, b_pixel);
            sumA += a_pixel;
            sumR += r_pixel;
            sumG += g_pixel;
            sumB += b_pixel;
            ++count;
        }
    }
    
    if (count > 0) {
        avgA = static_cast<int>(sumA / count);
        avgR = static_cast<int>(sumR / count);
        avgG = static_cast<int>(sumG / count);
        avgB = static_cast<int>(sumB / count);
    } else {
        avgA = avgR = avgG = avgB = 0;
    }
}

void Pixels::scale_to_bounding_box(const vec2& box, Pixels &scaled) const {
    // Calculate the scaling factor based on the bounding box
    float scale = min(static_cast<float>(box.x) / wh.x, static_cast<float>(box.y) / wh.y);

    // Calculate the new dimensions
    const ivec2 new_wh(wh.x * scale, wh.y * scale);

    // Scale the image using bicubic interpolation
    bicubic_scale(new_wh, scaled);
}

void Pixels::crop(const ivec2& top_left, const ivec2& bottom_right, Pixels &cropped) const {
    if(top_left.x < 0 || top_left.y < 0 || bottom_right.x > wh.x || bottom_right.y > wh.y || top_left.x >= bottom_right.x || top_left.y >= bottom_right.y)
        throw runtime_error("Crop dimensions out of range: " + to_string(top_left.x) + "," + to_string(top_left.y) + "," + to_string(bottom_right.x) + "," + to_string(bottom_right.y) + " for image of size " + to_string(wh.x) + "x" + to_string(wh.y));
    cropped = Pixels(bottom_right - top_left);
    for(int dx = top_left.x; dx < bottom_right.x; dx++) {
        int cropped_x = dx - top_left.x;
        for(int dy = top_left.y; dy < bottom_right.y; dy++)
            cropped.set_pixel_carelessly(cropped_x, dy - top_left.y, get_pixel_carefully(dx, dy));
    }
}

void Pixels::crop_by_alpha(Pixels& cropped) {
    int min_x = wh.x;
    int min_y = wh.y;
    int max_x = -1;
    int max_y = -1;

    // Find the bounding box of non-zero alpha pixels
    for (int y = 0; y < wh.y; y++) {
        for (int x = 0; x < wh.x; x++) {
            if (get_alpha(x, y) > 0) {
                min_x = min(min_x, x);
                min_y = min(min_y, y);
                max_x = max(max_x, x);
                max_y = max(max_y, y);
            }
        }
    }

    if(max_x >= min_x && max_y >= min_y) {
        crop(ivec2(min_x, min_y), ivec2(max_x + 1, max_y + 1), cropped);
    } else {
        cropped = Pixels(ivec2(0, 0));
    }
}

void Pixels::fill_rect(int x, int y, int rw, int rh, int col){
    if(x < 0) { rw += x; x = 0; }
    if(y < 0) { rh += y; y = 0; }
    if(x + rw > wh.x) rw = wh.x - x;
    if(y + rh > wh.y) rh = wh.y - y;
    for(int dx = 0; dx < rw; dx++)
        for(int dy = 0; dy < rh; dy++)
            set_pixel_carelessly(x+dx, y+dy, col);
}

Pixels Pixels::naive_scale_down(int scale_down_factor) const {
    Pixels result(wh/scale_down_factor);

    for (int y = 0; y*scale_down_factor < wh.y; y++) {
        for (int x = 0; x*scale_down_factor < wh.x; x++) {
            result.set_pixel_carelessly(x, y, get_pixel_carelessly(x*scale_down_factor, y*scale_down_factor));
        }
    }

    return result;
}

void Pixels::bicubic_scale(const ivec2& new_wh, Pixels& result) const {
    result = Pixels(new_wh);
    cuda_bicubic_scale(pixels.data(), wh, result.pixels.data(), new_wh);
}

void Pixels::print_to_terminal() {
    std::ostringstream ss;

    // Print an empty line first.
    ss << std::endl;

    // Get terminal dimensions.
    struct winsize wsz;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);
    const int termWidth = wsz.ws_col;
    // Note: we are not using termHeight here.

    // Assume a half-block is square
    const double charAspect = 2.0;

    int outputWidth = min(wh.x, termWidth);
    double imageAspect = static_cast<double>(wh.x) / wh.y;

    // Determine the effective vertical resolution (in image pixels) that we wish to sample.
    // We multiply by 2 because each printed line represents two image rows.
    int sample_height = static_cast<int>(outputWidth / (imageAspect * charAspect) * 2);
    // Ensure sample_height is even.
    sample_height -= sample_height % 2;
    int printedLines = sample_height / 2;

    // For each terminal character cell, we determine the corresponding region in the image.
    for (int y = 0; y < printedLines; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            // Determine the horizontal region that maps to this terminal column.
            int x0 = x * wh.x / outputWidth;
            int x1 = (x + 1) * wh.x / outputWidth;

            // For the top half of the character cell:
            int top_y0 = (2 * y) * wh.y / sample_height;
            int top_y1 = (2 * y + 1) * wh.y / sample_height;

            // For the bottom half of the character cell:
            int bot_y0 = (2 * y + 1) * wh.y / sample_height;
            int bot_y1 = (2 * y + 2) * wh.y / sample_height;

            int a_top, r_top, g_top, b_top;
            int a_bot, r_bot, g_bot, b_bot;

            // Supersample (average) over the regions.
            get_average_color(x0, top_y0, x1, top_y1, a_top, r_top, g_top, b_top);
            get_average_color(x0, bot_y0, x1, bot_y1, a_bot, r_bot, g_bot, b_bot);

            // Use ANSI true-color escape sequences:
            //  - Set foreground to the average top color.
            //  - Set background to the average bottom color.
            // Then print the Unicode upper half block (▀), which renders the top half in
            // the foreground color and the bottom half in the background color.
            ss << "\033[38;2;" << r_top << ";" << g_top << ";" << b_top << "m"
                 << "\033[48;2;" << r_bot << ";" << g_bot << ";" << b_bot << "m"
                 << reinterpret_cast<const char*>(u8"\u2580");
        }
        // Reset colors at the end of each line.
        ss << "\033[0m" << std::endl;
    }
    // Reset at the end.
    ss << "\033[0m" << std::endl;

    std::cout << ss.str();
}
