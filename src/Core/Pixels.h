#pragma once

#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <fstream>
#include <sstream>
#include <stack>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include "../Host_Device_Shared/helpers.h"
#include "Color.cpp"

using namespace std;

inline int OPAQUE_BLACK = 0xFF000000;
inline int OPAQUE_WHITE = 0xFFFFFFFF;
inline int TRANSPARENT_BLACK = 0x00000000;
inline int TRANSPARENT_WHITE = 0x00FFFFFF;

class Pixels{
public:
    int w;
    int h;
    vector<unsigned int> pixels;
    Pixels() : w(0), h(0), pixels(0){};
    Pixels(int width, int height) : w(width), h(height), pixels(width*height){};

    inline bool out_of_range(int x, int y) const {
        return x < 0 || x >= w || y < 0 || y >= h;
    }

    inline int get_pixel_carelessly(int x, int y) const {
        // TODO Remove before flight
        if(out_of_range(x, y)) throw runtime_error("Pixel access out of range.");
        return pixels[w*y+x];
    }

    inline int get_pixel_carefully(int x, int y) const {
        if(out_of_range(x, y)) return TRANSPARENT_BLACK; // 0
        return pixels[w*y+x];
    }

    inline void overlay_pixel(int x, int y, int col, double overlay_opacity_multiplier = 1){
        set_pixel_carefully(x, y, color_combine(get_pixel_carefully(x, y), col, overlay_opacity_multiplier));
    }

    inline void get_pixel_by_channels(int x, int y, int& a, int& r, int& g, int& b) const {
        int col = get_pixel_carefully(x,y);
        a=geta(col);
        r=getr(col);
        g=getg(col);
        b=getb(col);
    }

    inline int get_alpha(int x, int y) const {
        return geta(get_pixel_carefully(x,y));
    }

    inline void set_pixel_carelessly(int x, int y, int col) {
        // TODO Remove before flight
        if(out_of_range(x, y)) throw runtime_error("Pixel set out of range.");
        pixels[w*y+x] = col;
    }

    inline void set_pixel_carefully(int x, int y, int col) {
        if(out_of_range(x, y)) return;
        pixels[w*y+x] = col;
    }

    void darken(float factor){
        for(int i = 0; i < w*h; i++){
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

    inline void set_alpha(int x, int y, int a) {
        if(out_of_range(x, y) || a < 0 || a > 255) return;
        pixels[w*y+x] = (pixels[w*y+x] & 0x00ffffff) | (a << 24);
    }

    // ---------------------------------------------------------------------
    // get_average_color:
    // Given a rectangle from (x_start, y_start) up to (but not including) (x_end, y_end),
    // average the pixel colors in that region. The averages for the R, G, and B channels
    // are returned via the reference parameters.
    void get_average_color(int x_start, int y_start, int x_end, int y_end,
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

    void scale_to_bounding_box(int box_w, int box_h, Pixels &scaled) const {
        // Calculate the scaling factor based on the bounding box
        float scale = min(static_cast<float>(box_w) / w, static_cast<float>(box_h) / h);

        // Calculate the new dimensions
        int new_width = static_cast<int>(w * scale);
        int new_height = static_cast<int>(h * scale);

        // Scale the image using bicubic interpolation
        bicubic_scale(new_width, new_height, scaled);
    }

    void crop(int x, int y, int cw, int ch, Pixels &cropped) const {
        if(x < 0 || y < 0 || x + cw > w || y + ch > h)
            throw runtime_error("Crop dimensions out of range: " + to_string(x) + "," + to_string(y) + "," + to_string(cw) + "," + to_string(ch) + " for image of size " + to_string(w) + "x" + to_string(h));
        cropped = Pixels(cw, ch);
        for(int dx = 0; dx < cw; dx++)
            for(int dy = 0; dy < ch; dy++)
                cropped.set_pixel_carelessly(dx, dy, get_pixel_carelessly(x+dx, y+dy));
    }

    void crop_by_fractions(float crop_top, float crop_bottom, float crop_left, float crop_right, Pixels &cropped) const {
        int x =  w * crop_left;
        int y =  h * crop_top;
        int cw =  w * (1.0f - crop_left - crop_right);
        int ch =  h * (1.0f - crop_top - crop_bottom);
        crop(x, y, cw, ch, cropped);
    }

    bool is_empty() const {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (get_alpha(x, y) != 0) {
                    return false; // Found a pixel with non-zero alpha, so the Pixels is not empty
                }
            }
        }
        return true; // No pixel with non-zero alpha found, Pixels is empty
    }

    void add_border(int col, int thickness = 1){
        if(thickness > w || thickness > h)
            throw runtime_error("Border thickness too large.");
        for(int t = 0; t < thickness; t++){
            for(int x = 0; x < w; x++){
                set_pixel_carelessly(x, t, col);
                set_pixel_carelessly(x, h-1-t, col);
            }
            for(int y = 0; y < h; y++){
                set_pixel_carelessly(t, y, col);
                set_pixel_carelessly(w-1-t, y, col);
            }
        }
    }

    // TODO this function is already written in CUDA. use that instead.
    void overlay(Pixels p, int dx, int dy, double overlay_opacity_multiplier = 1){
        for(int x = 0; x < p.w; x++){
            int xpdx = x+dx;
            for(int y = 0; y < p.h; y++){
                overlay_pixel(xpdx, y+dy, p.get_pixel_carefully(x, y), overlay_opacity_multiplier);
            }
        }
    }

    void overwrite(Pixels p, int dx, int dy){
        for(int x = 0; x < p.w; x++){
            int xpdx = x+dx;
            for(int y = 0; y < p.h; y++){
                set_pixel_carefully(xpdx, y+dy, p.get_pixel_carefully(x, y));
            }
        }
    }

    void fill_rect(int x, int y, int rw, int rh, int col){
        if(x < 0) { rw += x; x = 0; }
        if(y < 0) { rh += y; y = 0; }
        if(x + rw > w) rw = w - x;
        if(y + rh > h) rh = h - y;
        for(int dx = 0; dx < rw; dx++)
            for(int dy = 0; dy < rh; dy++)
                set_pixel_carelessly(x+dx, y+dy, col);
    }

    void fill_circle(double x, double y, double r, int col, double opa=1){
        fill_ring(x, y, r, 0, col, opa);
    }

    void fill_ring(double x, double y, double r_outer, double r_inner, int col, double opa=1){
        double r_outer_sq = square(r_outer);
        double r_inner_sq = square(r_inner);
        for(double dx = -r_outer+1; dx < r_outer; dx++){
            double sdx = square(dx);
            for(double dy = -r_outer+1; dy < r_outer; dy++) {
                double sdy = square(dy);
                if(sdx+sdy < r_outer_sq && sdx+sdy >= r_inner_sq)
                    overlay_pixel(x+dx, y+dy, col, opa);
            }
        }
    }

    void fill_ellipse(double x, double y, double rw, double rh, int col, double opa=1){
        for(double dx = -rw+1; dx < rw; dx++){
            double sdx = square(dx/rw);
            for(double dy = -rh+1; dy < rh; dy++)
                if(sdx+square(dy/rh) < 1)
                    overlay_pixel(x+dx, y+dy, col, opa);
        }
    }

    void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness) {
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

    // The x, y, rw, rh parameters here correspond to the external bounding box.
    void rounded_rect(float x, float y, float rw, float rh, float r, int col){
        int xplusr = round(x+r);
        int yplusr = round(y+r);
        int rwmrt2 = round(rw-r*2);
        int rhmrt2 = round(rh-r*2);
        int xprwmr = round(x+rw-r);
        int yprhmr = round(y+rh-r);
        fill_rect(xplusr, y, rwmrt2, rh, col);
        fill_rect(x, yplusr, rw, rhmrt2, col);
        for(int i = 0; i < 4; i++)
            fill_ellipse(i%2==0 ? xplusr : xprwmr, i/2==0 ? yplusr : yprhmr, r, r, col);
    }

    void flood_fill(int x, int y, int color) {
        int targetColor = get_pixel_carefully(x, y);

        // Check if the target pixel already has the desired color
        if (targetColor == color)
            return;

        std::stack<std::pair<int, int>> stack;
        stack.push({x, y});

        while (!stack.empty()) {
            auto [curX, curY] = stack.top();
            stack.pop();

            // Check if current pixel is within the image boundaries
            if (out_of_range(curX, curY))
                continue;

            // Check if current pixel is the target color
            if (get_pixel_carelessly(curX, curY) != targetColor)
                continue;

            // Update the pixel color
            set_pixel_carelessly(curX, curY, color);

            // Push neighboring pixels onto the stack
            stack.push({curX - 1, curY}); // Left
            stack.push({curX + 1, curY}); // Right
            stack.push({curX, curY - 1}); // Up
            stack.push({curX, curY + 1}); // Down
        }
    }

    Pixels naive_scale_down(int scale_down_factor) const {
        Pixels result(w/scale_down_factor, h/scale_down_factor);

        for (int y = 0; y*scale_down_factor < h; y++) {
            for (int x = 0; x*scale_down_factor < w; x++) {
                result.set_pixel_carelessly(x, y, get_pixel_carelessly(x*scale_down_factor, y*scale_down_factor));
            }
        }

        return result;
    }

    void rotate_90_inverse(Pixels& rotated) const {
        rotated = Pixels(h, w);

        // Map each pixel to its new location
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                rotated.set_pixel_carelessly(y, w-1-x, get_pixel_carelessly(x, y));
            }
        }
    }

    void rotate_90(Pixels& rotated) const {
        rotated = Pixels(h, w);

        // Map each pixel to its new location
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                rotated.set_pixel_carelessly(h-1-y, x, get_pixel_carelessly(x, y));
            }
        }
    }

    void bicubic_scale(int new_width, int new_height, Pixels& result) const {
        result = Pixels(new_width, new_height);

        float x_ratio = static_cast<float>(w) / new_width;
        float y_ratio = static_cast<float>(h) / new_height;

        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                float gx = x * x_ratio;
                float gy = y * y_ratio;

                int gxi = static_cast<int>(gx);
                int gyi = static_cast<int>(gy);

                float dx = gx - gxi;
                float dy = gy - gyi;

                float pa = 0;
                float pr = 0;
                float pg = 0;
                float pb = 0;

                // Iterate over the surrounding 4x4 block of pixels
                for (int m = -1; m <= 2; m++) {
                    for (int n = -1; n <= 2; n++) {
                        int xi = gxi + m;
                        int yi = gyi + n;

                        xi = std::clamp(xi, 0, w - 1);
                        yi = std::clamp(yi, 0, h - 1);

                        int pixel = get_pixel_carefully(xi, yi);
                        float weight = bicubic_weight(dx - m) * bicubic_weight(dy - n);

                        pa += weight * geta(pixel);
                        pr += weight * getr(pixel);
                        pg += weight * getg(pixel);
                        pb += weight * getb(pixel);
                    }
                }
                pa = std::clamp(static_cast<int>(pa), 0, 255);
                pr = std::clamp(static_cast<int>(pr), 0, 255);
                pg = std::clamp(static_cast<int>(pg), 0, 255);
                pb = std::clamp(static_cast<int>(pb), 0, 255);

                result.set_pixel_carelessly(x, y, argb(pa, pr, pg, pb));
            }
        }
    }

    // Bicubic weight function
    float bicubic_weight(float t) const {
        const float a = -0.5f; // Commonly used value for bicubic interpolation

        if (t < 0) t = -t;
        float t2 = t * t;
        float t3 = t2 * t;

        if (t <= 1) {
            return (a + 2) * t3 - (a + 3) * t2 + 1;
        } else if (t < 2) {
            return a * (t3 - 5 * t2 + 8 * t - 4);
        } else {
            return 0;
        }
    }

    // ---------------------------------------------------------------------
    // This function outputs the image to the terminal using Unicode half-block
    // characters (▀). Each printed character cell represents two rows of supersampled
    // image data. The top half of the block uses the foreground color and the bottom
    // half uses the background color.
    //
    // We compute the sampling regions by mapping the terminal grid to the source image
    // and then average over each corresponding rectangle.
    void print_to_terminal() {
        // Print an empty line first.
        cout << endl;

        // Get terminal dimensions.
        struct winsize wsz;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);
        const int termWidth = wsz.ws_col;
        // Note: we are not using termHeight here.

        // Assume a half-block is square
        const double charAspect = 2.0;

        int outputWidth = min(w, termWidth);
        double imageAspect = static_cast<double>(w) / h;

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
                int x0 = x * w / outputWidth;
                int x1 = (x + 1) * w / outputWidth;

                // For the top half of the character cell:
                int top_y0 = (2 * y) * h / sample_height;
                int top_y1 = (2 * y + 1) * h / sample_height;

                // For the bottom half of the character cell:
                int bot_y0 = (2 * y + 1) * h / sample_height;
                int bot_y1 = (2 * y + 2) * h / sample_height;

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
};

Pixels create_alpha_from_intensities(const vector<vector<unsigned int>>& intensities) {
    int height = intensities.size();
    int width = (height > 0) ? intensities[0].size() : 0;

    Pixels result(width, height);

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

Pixels create_pixels_from_2d_vector(const vector<vector<unsigned int>>& colors, int negative_intensity) {
    int height = colors.size();
    int width = (height > 0) ? colors[0].size() : 0;

    Pixels result(width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            result.pixels[y+x*width] = colors[y][x];
        }
    }

    return result;
}

Pixels crop_by_alpha(const Pixels& p) {
    int min_x = p.w;
    int min_y = p.h;
    int max_x = -1;
    int max_y = -1;

    // Find the bounding box of non-zero alpha pixels
    for (int y = 0; y < p.h; y++) {
        for (int x = 0; x < p.w; x++) {
            if (p.get_alpha(x, y) > 0) {
                min_x = min(min_x, x);
                min_y = min(min_y, y);
                max_x = max(max_x, x);
                max_y = max(max_y, y);
            }
        }
    }

    // Calculate the dimensions of the cropped Pixels
    int width = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    // Create the cropped Pixels object
    Pixels cropped_pixels(width, height);

    // Copy the pixels within the bounding box to the cropped Pixels
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            cropped_pixels.set_pixel_carelessly(x - min_x, y - min_y, p.get_pixel_carelessly(x, y));
        }
    }

    return cropped_pixels;
}
