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

extern "C" int cuda_bicubic_scale(const unsigned int* input_pixels, const vec2& input_size, unsigned int* output_pixels, const vec2& output_size);

Pixels::Pixels() : size(0,0), pixels(0) {}
Pixels::Pixels(const vec2& dim) : size(vec2((int) dim.x, (int) dim.y)), pixels((int)dim.x * (int)dim.y) {}

bool Pixels::out_of_range(int x, int y) const {
    return x < 0 || x >= size.x || y < 0 || y >= size.y;
}

int Pixels::get_pixel_carelessly(int x, int y) const {
    return pixels[size.x*y+x];
}

int Pixels::get_pixel_carefully(int x, int y) const {
    if(out_of_range(x, y)) return TRANSPARENT_BLACK; // 0
    return pixels[size.x*y+x];
}

void Pixels::overlay_pixel(int x, int y, int col, double overlay_opacity_multiplier){
    set_pixel_carefully(x, y, color_combine(get_pixel_carefully(x, y), col, overlay_opacity_multiplier));
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
    pixels[size.x*y+x] = col;
}

void Pixels::set_pixel_carefully(int x, int y, int col) {
    if(out_of_range(x, y)) return;
    pixels[size.x*y+x] = col;
}

void Pixels::darken(float factor){
    for(int i = 0; i < (int)(size.x)*(int)(size.y); i++){
        int a = geta(pixels[i]);
        int r = getr(pixels[i]);
        int g = getg(pixels[i]);
        int b = getb(pixels[i]);
        r = static_cast<int>(r * factor);
        g = static_cast<int>(g * factor);
        b = static_cast<int>(b * factor);
        pixels[i] = argb(a, r, g, b);
    }
}

void Pixels::set_alpha(int x, int y, int a) {
    if(out_of_range(x, y) || a < 0 || a > 255) return;
    pixels[size.x*y+x] = (pixels[size.x*y+x] & 0x00ffffff) | (a << 24);
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

void Pixels::scale_to_bounding_box(const vec2& box_dim, Pixels &scaled) const {
    // Calculate the scaling factor based on the bounding box
    const vec2 div = box_dim / size;
    float scale = min(div.x, div.y);

    // Calculate the new dimensions
    const vec2 new_dim = scale * size;

    // Scale the image using bicubic interpolation
    bicubic_scale(new_dim, scaled);
}

void Pixels::crop(const vec2& top_left, const vec2& cropped_dimensions, Pixels &cropped) const {
    if(top_left.x < 0 || top_left.y < 0 || top_left.x + cropped_dimensions.x > size.x || top_left.y + cropped_dimensions.y > size.y)
        throw runtime_error("Crop dimensions out of range: " + to_string(top_left.x) + "," + to_string(top_left.y) + "," + to_string(cropped_dimensions.x) + "," + to_string(cropped_dimensions.y) + " for image of size " + to_string(size.x) + "x" + to_string(size.y));
    cropped = Pixels(cropped_dimensions);
    for(int dx = 0; dx < cropped_dimensions.x; dx++)
        for(int dy = 0; dy < cropped_dimensions.y; dy++)
            cropped.set_pixel_carelessly(dx, dy, get_pixel_carelessly(top_left.x+dx, top_left.y+dy));
}

void Pixels::crop_by_fractions(const vec2& crop_top_left, const vec2& crop_bottom_right, Pixels &cropped) const {
    const vec2 tl_pos = size * crop_top_left;
    const vec2 dim = size * (vec2(1,1) - crop_bottom_right - crop_top_left);
    crop(tl_pos, dim, cropped);
}

int Pixels::get_pixel_bilinear(double x, double y) const {
    int x0 = static_cast<int>(floor(x));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(floor(y));
    int y1 = y0 + 1;

    double dx = x - x0;
    double dy = y - y0;

    int c00 = get_pixel_carefully(x0, y0);
    int c10 = get_pixel_carefully(x1, y0);
    int c01 = get_pixel_carefully(x0, y1);
    int c11 = get_pixel_carefully(x1, y1);

    int a00 = geta(c00), r00 = getr(c00), g00 = getg(c00), b00 = getb(c00);
    int a10 = geta(c10), r10 = getr(c10), g10 = getg(c10), b10 = getb(c10);
    int a01 = geta(c01), r01 = getr(c01), g01 = getg(c01), b01 = getb(c01);
    int a11 = geta(c11), r11 = getr(c11), g11 = getg(c11), b11 = getb(c11);

    int a0 = static_cast<int>(a00 * (1 - dx) + a10 * dx);
    int r0 = static_cast<int>(r00 * (1 - dx) + r10 * dx);
    int g0 = static_cast<int>(g00 * (1 - dx) + g10 * dx);
    int b0 = static_cast<int>(b00 * (1 - dx) + b10 * dx);

    int a1 = static_cast<int>(a01 * (1 - dx) + a11 * dx);
    int r1 = static_cast<int>(r01 * (1 - dx) + r11 * dx);
    int g1 = static_cast<int>(g01 * (1 - dx) + g11 * dx);
    int b1 = static_cast<int>(b01 * (1 - dx) + b11 * dx);

    int a = static_cast<int>(a0 * (1 - dy) + a1 * dy);
    int r = static_cast<int>(r0 * (1 - dy) + r1 * dy);
    int g = static_cast<int>(g0 * (1 - dy) + g1 * dy);
    int b = static_cast<int>(b0 * (1 - dy) + b1 * dy);
    return argb(a, r, g, b);
}

void Pixels::overlay(const Pixels& p, const vec2& pos, double overlay_opacity_multiplier){
    int dx = pos.x;
    int dy = pos.y;
    for(int x = 0; x < p.size.x; x++){
        int xpdx = x+dx;
        for(int y = 0; y < p.size.y; y++){
            overlay_pixel(xpdx, y+dy, p.get_pixel_carefully(x, y), overlay_opacity_multiplier);
        }
    }
}

void Pixels::overwrite(const Pixels& p, const vec2& pos) {
    int dx = pos.x;
    int dy = pos.y;
    for(int x = 0; x < p.size.x; x++){
        int xpdx = x+dx;
        for(int y = 0; y < p.size.y; y++){
            set_pixel_carefully(xpdx, y+dy, p.get_pixel_carefully(x, y));
        }
    }
}

void Pixels::fill_rect(const vec2& pos, const vec2& dimensions, int col){
    int x = pos.x;
    int y = pos.y;
    int rw = dimensions.x;
    int rh = dimensions.y;
    if(x < 0) { rw += x; x = 0; }
    if(y < 0) { rh += y; y = 0; }
    if(x + rw > size.x) rw = size.x - x;
    if(y + rh > size.y) rh = size.y - y;
    for(int dx = 0; dx < rw; dx++)
        for(int dy = 0; dy < rh; dy++)
            set_pixel_carelessly(x+dx, y+dy, col);
}

void Pixels::fill_circle(const vec2& pos, double r, int col, double opa){
    fill_ring(pos, r, 0, col, opa);
}

void Pixels::fill_ring(const vec2& pos, double r_outer, double r_inner, int col, double opa){
    const int x = pos.x;
    const int y = pos.y;
    const double r_outer_sq = square(r_outer);
    const double r_inner_sq = square(r_inner);
    for(double dx = -r_outer+1; dx < r_outer; dx++){
        const double sdx = square(dx);
        for(double dy = -r_outer+1; dy < r_outer; dy++) {
            const double sdx_sdy = sdx + square(dy);
            if(sdx_sdy < r_outer_sq && sdx_sdy >= r_inner_sq)
                overlay_pixel(x+dx, y+dy, col, opa);
        }
    }
}

void Pixels::fill_ellipse(const vec2& pos, const vec2& dimensions, int col, double opa){
    const int x = pos.x;
    const int y = pos.y;
    const int rw = dimensions.x;
    const int rh = dimensions.y;
    for(double dx = -rw+1; dx < rw; dx++){
        const double sdx = square(dx/rw);
        for(double dy = -rh+1; dy < rh; dy++)
            if(sdx+square(dy/rh) < 1)
                overlay_pixel(x+dx, y+dy, col, opa);
    }
}

void Pixels::bresenham(const vec2& start, const vec2& end, int col, float opacity, int thickness) {
    int x1 = start.x;
    int y1 = start.y;
    int x2 = end.x;
    int y2 = end.y;
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    if(dx > 10000 || dy > 10000){
        if (false) {
            cout << "Bresenham Overflow. Quitting without finishing line." << endl;
            cout << "x1: " << x1 << endl;
            cout << "y1: " << y1 << endl;
            cout << "x2: " << x2 << endl;
            cout << "y2: " << y2 << endl;
            cout << "col: " << col << endl;
            cout << "thickness: " << thickness << endl;
        }
        return;
    }

    int sx = (x1 < x2) ? 1 : -1; // Direction of drawing on x axis
    int sy = (y1 < y2) ? 1 : -1; // Direction of drawing on y axis

    int err = dx - dy;

    while(true) {
        overlay_pixel(x1, y1, col, opacity);
        for(int i = 1; i < thickness; i++){
            overlay_pixel(x1+i, y1  , col, opacity);
            overlay_pixel(x1-i, y1  , col, opacity);
            overlay_pixel(x1  , y1+i, col, opacity);
            overlay_pixel(x1  , y1-i, col, opacity);
        }

        // If we've reached the end point, break
        if (x1 == x2 && y1 == y2) break;

        int e2 = err * 2;

        // Adjust for next point based on the error
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

void Pixels::rounded_rect(const vec2& pos, const vec2& dimensions, float r, int col){
    const vec2 posplusr = pos + vec2(r,r);
    const vec2 dimm2r = dimensions - vec2(2*r, 2*r);
    const vec2 xpdimmr = pos + dimensions - vec2(r,r);
    fill_rect(vec2(posplusr.x, pos.y), vec2(dimm2r.x, dimensions.y), col);
    fill_rect(vec2(pos.x, posplusr.y), vec2(dimensions.x, dimm2r.y), col);
    for(int i = 0; i < 4; i++)
        fill_circle(vec2(i%2==0 ? posplusr.x : xpdimmr.x, i/2==0 ? posplusr.y : xpdimmr.y), r, col);
}

Pixels Pixels::naive_scale_down(int scale_down_factor) const {
    Pixels result(size/scale_down_factor);

    for (int y = 0; y*scale_down_factor < size.y; y++) {
        for (int x = 0; x*scale_down_factor < size.x; x++) {
            result.set_pixel_carelessly(x, y, get_pixel_carelessly(x*scale_down_factor, y*scale_down_factor));
        }
    }

    return result;
}

void Pixels::bicubic_scale(const vec2& new_dim, Pixels& result) const {
    result = Pixels(new_dim);
    cuda_bicubic_scale(pixels.data(), size, result.pixels.data(), new_dim);
}

void Pixels::print_to_terminal() {
    // Print an empty line first.
    cout << endl;

    // Get terminal dimensions.
    struct winsize wsz;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);
    const int termWidth = wsz.ws_col;
    // Note: we are not using termHeight here.

    // Assume a half-block is square
    const double charAspect = 2.0;

    int outputWidth = min((int)size.x, termWidth);
    double imageAspect = static_cast<double>(size.x) / size.y;

    // Determine the effective vertical resolution (in image pixels) that we wish to sample.
    // We multiply by 2 because each printed line represents two image rows.
    int sample_height = static_cast<int>(outputWidth / (imageAspect * charAspect) * 2);
    // Ensure sample_height is even.
    sample_height -= sample_height % 2;
    int printedLines = sample_height / 2;

    int VIDEO_BACKGROUND_COLOR = get_video_background_color();

    // For each terminal character cell, we determine the corresponding region in the image.
    for (int y = 0; y < printedLines; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            // Determine the horizontal region that maps to this terminal column.
            int x0 = x * size.x / outputWidth;
            int x1 = (x + 1) * size.x / outputWidth;

            // For the top half of the character cell:
            int top_y0 = (2 * y) * size.y / sample_height;
            int top_y1 = (2 * y + 1) * size.y / sample_height;

            // For the bottom half of the character cell:
            int bot_y0 = (2 * y + 1) * size.y / sample_height;
            int bot_y1 = (2 * y + 2) * size.y / sample_height;

            int a_top, r_top, g_top, b_top;
            int a_bot, r_bot, g_bot, b_bot;

            // Supersample (average) over the regions.
            get_average_color(x0, top_y0, x1, top_y1, a_top, r_top, g_top, b_top);
            get_average_color(x0, bot_y0, x1, bot_y1, a_bot, r_bot, g_bot, b_bot);

            double alpha_top = a_top / 255.;
            double one_minus_alpha_top = 1-alpha_top;
            r_top = r_top * alpha_top + getr(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_top;
            g_top = g_top * alpha_top + getg(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_top;
            b_top = b_top * alpha_top + getb(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_top;

            double alpha_bot = a_bot / 255.;
            double one_minus_alpha_bot = 1-alpha_bot;
            r_bot = r_bot * alpha_bot + getr(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_bot;
            g_bot = g_bot * alpha_bot + getg(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_bot;
            b_bot = b_bot * alpha_bot + getb(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_bot;

            // Use ANSI true-color escape sequences:
            //  - Set foreground to the average top color.
            //  - Set background to the average bottom color.
            // Then print the Unicode upper half block (▀), which renders the top half in
            // the foreground color and the bottom half in the background color.
            cout << "\033[38;2;" << r_top << ";" << g_top << ";" << b_top << "m"
                 << "\033[48;2;" << r_bot << ";" << g_bot << ";" << b_bot << "m"
                 << "\u2580";
        }
        // Reset colors at the end of each line.
        cout << "\033[0m" << endl;
    }
    // Reset at the end.
    cout << "\033[0m" << endl;
}

// Free functions implementations

Pixels create_alpha_from_intensities(const vector<vector<unsigned int>>& intensities) {
    int height = intensities.size();
    int width = (height > 0) ? intensities[0].size() : 0;

    Pixels result(vec2(width, height));

    // Find the minimum and maximum intensity values
    unsigned int minIntensity = numeric_limits<unsigned int>::max();
    unsigned int maxIntensity = numeric_limits<unsigned int>::min();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned int intensity = intensities[y][x];
            if(intensity < 0) continue;
            minIntensity = min(minIntensity, intensity);
            maxIntensity = max(maxIntensity, intensity);
        }
    }

    // Calculate the range of intensities
    int intensityRange = maxIntensity - minIntensity;
    if (intensityRange == 0) {
        // Avoid division by zero if all intensities are the same
        intensityRange = 1;
    }

    // Set the alpha channel based on normalized intensity values
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned int intensity = intensities[y][x];
            // Square it to make it more peaky
            int normalized_squared_intensity = square(intensity - minIntensity) * 255. / square(intensityRange);
            result.set_pixel_carefully(x, y, (normalized_squared_intensity << 24) | 0x00FFFFFF); // White color with alpha
        }
    }

    return result;
}

Pixels crop_by_alpha(const Pixels& p) {
    vec2 min_pos(p.size);
    vec2 max_pos(-1,-1);

    // Find the bounding box of non-zero alpha pixels
    for (int y = 0; y < p.size.y; y++) {
        for (int x = 0; x < p.size.x; x++) {
            if (p.get_alpha(x, y) > 0) {
                const vec2 pos(x,y);
                min_pos = vec_min(min_pos, pos);
                max_pos = vec_max(max_pos, pos);
            }
        }
    }

    // Calculate the dimensions of the cropped Pixels
    const vec2 cropped_dim(max_pos - min_pos + vec2(1,1));

    // Create the cropped Pixels object
    Pixels cropped_pixels(cropped_dim);

    // Copy the pixels within the bounding box to the cropped Pixels
    for (int y = min_pos.y; y <= max_pos.y; y++) {
        for (int x = min_pos.x; x <= max_pos.x; x++) {
            cropped_pixels.set_pixel_carelessly(x - min_pos.x, y - min_pos.y, p.get_pixel_carelessly(x, y));
        }
    }

    return cropped_pixels;
}
