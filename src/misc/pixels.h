#pragma once

#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <fstream>
#include <sstream>
#include <stack>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include "inlines.h"
#include "color.cpp"

using namespace std;

inline int OPAQUE_BLACK = 0xFF000000;
inline int OPAQUE_WHITE = 0xFFFFFFFF;
inline int TRANSPARENT_BLACK = 0x00000000;
inline int TRANSPARENT_WHITE = 0x00FFFFFF;

class Pixels{
public:
    int w;
    int h;
    vector<int> pixels;
    Pixels() : w(0), h(0), pixels(0){};
    Pixels(int width, int height) : w(width), h(height), pixels(width*height){};
    // Copy constructor
    Pixels(const Pixels& other) : w(other.w), h(other.h), pixels(other.pixels) {};

    inline bool out_of_range(int x, int y) const {
        return x < 0 || x >= w || y < 0 || y >= h;
    }

    inline int get_pixel(int x, int y) const {
        if(out_of_range(x, y)) return 0;
        return pixels[w*y+x];
    }

    inline void overlay_pixel(int x, int y, int col, double overlay_opacity_multiplier = 1){
        set_pixel(x, y, color_combine(get_pixel(x, y), col, overlay_opacity_multiplier));
    }

    inline void get_pixel_by_channels(int x, int y, int& a, int& r, int& g, int& b) const {
        int col = get_pixel(x,y);
        a=geta(col);
        r=getr(col);
        g=getg(col);
        b=getb(col);
    }

    inline int get_alpha(int x, int y) const {
        return geta(get_pixel(x,y));
    }

    inline void set_pixel(int x, int y, int col) {
        if(out_of_range(x, y)) return;
        pixels[w*y+x] = col;
    }

    inline void set_alpha(int x, int y, int a) {
        if(out_of_range(x, y) || a < 0 || a > 255) return;
        pixels[w*y+x] = (pixels[w*y+x] & 0x00ffffff) | (a << 24);
    }

    void print_to_terminal() const {
        cout << endl;
        struct winsize wsz;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);

        const int twidth = wsz.ws_col;
        const int theight = wsz.ws_row;
        const int width = min(twidth, theight*w/h);
        const int height = min(theight, twidth*h/w);
        const int xStep = max(w / width, 1);
        const int yStep = max(h / height, 1);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int sampleX = x * xStep / 3 + w / 3;
                int sampleY = y * yStep;

                int a, r, g, b;
                //there is something weird going on here, dont know why the params have to be in reverse
                get_pixel_by_channels(sampleX, sampleY, a, b, g, r);

                // Map the RGB values to ANSI color codes
                int rCode = static_cast<int>((1-cube(1-(r / 255.0*a/255.))) * 5);
                int gCode = static_cast<int>((1-cube(1-(g / 255.0*a/255.))) * 5);
                int bCode = static_cast<int>((1-cube(1-(b / 255.0*a/255.))) * 5);

                // Calculate the ANSI color code based on RGB values
                int colorCode = 16 + 36 * bCode + 6 * gCode + rCode;

                // Output the colored ASCII character
                cout << "\033[48;5;" << colorCode << "m";
                cout << ' ';
            }
            cout << "\033[0m" << endl;
        }
        cout << endl;
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
        for(int t = 0; t < thickness; t++){
            for(int x = 0; x < w; x++){
                set_pixel(x, t, col);
                set_pixel(x, h-1-t, col);
            }
            for(int y = 0; y < h; y++){
                set_pixel(t, y, col);
                set_pixel(w-1-t, y, col);
            }
        }
    }

    void overlay(Pixels p, int dx, int dy, double overlay_opacity_multiplier = 1){
        for(int x = 0; x < p.w; x++){
            int xpdx = x+dx;
            for(int y = 0; y < p.h; y++){
                overlay_pixel(xpdx, y+dy, p.get_pixel(x, y), overlay_opacity_multiplier);
            }
        }
    }

    void underlay(Pixels p, int dx, int dy){
        for(int x = 0; x < p.w; x++){
            int xpdx = x+dx;
            for(int y = 0; y < p.h; y++){
                int col = color_combine(p.get_pixel(x, y), get_pixel(xpdx, y+dy));
                set_pixel(xpdx, y+dy, col);
            }
        }
    }

    void overwrite(Pixels p, int dx, int dy){
        for(int x = 0; x < p.w; x++){
            int xpdx = x+dx;
            for(int y = 0; y < p.h; y++){
                set_pixel(xpdx, y+dy, p.get_pixel(x, y));
            }
        }
    }

    void invert(){
        for(int i = 0; i < pixels.size(); i++)
            if(i%4!=3)
                pixels[i] = 255-pixels[i];
    }

    void fade(double f){
        for(int i = 0; i < pixels.size(); i++)
            pixels[i] *= f;
    }

    void grayscale_to_alpha(){
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int a, r, g, b;
                get_pixel_by_channels(x, y, a, r, g, b);
                if(r != b || g != r) continue; // if this pixel is not true grayscale
                pixels[x+y*w] = argb_to_col(r, 255, 255, 255);
            }
        }
    }

    void fill_rect(int x, int y, int rw, int rh, int col){
        for(int dx = 0; dx < rw; dx++)
            for(int dy = 0; dy < rh; dy++)
                set_pixel(x+dx, y+dy, col);
    }

    void fill_circle(double x, double y, double r, int col){
        fill_ellipse(x, y, r, r, col);
    }

    void fill_ellipse(double x, double y, double rw, double rh, int col){
        for(double dx = -rw+1; dx < rw; dx++)
            for(double dy = -rh+1; dy < rh; dy++)
                if(square(dx/rw)+square(dy/rh) < 1)
                    set_pixel(x+dx, y+dy, col);
    }

    void fill(int col){
        fill_rect(0, 0, w, h, col);
    }

    void bitwise_and(int bitstrip){
        for(int x = 0; x < w; x++)
            for(int y = 0; y < h; y++)
                pixels[x+y*w] &= bitstrip;
    }

    void bresenham(int x1, int y1, int x2, int y2, int col, int thickness) {
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);

        int sx = (x1 < x2) ? 1 : -1; // Direction of drawing on x axis
        int sy = (y1 < y2) ? 1 : -1; // Direction of drawing on y axis

        int err = dx - dy;

        int count = 0;
        while(true) {
            count++;
            if(count > 1000000){
                cout << "Bresenham Overflow. Quitting without finishing line." << endl;
                cout << "x1: " << x1 << endl;
                cout << "y1: " << y1 << endl;
                cout << "x2: " << x2 << endl;
                cout << "y2: " << y2 << endl;
                cout << "col: " << col << endl;
                cout << "thickness: " << thickness << endl;
                return;
            }
            set_pixel(x1, y1, col);
            for(int i = 1; i < thickness; i++){
                set_pixel(x1+i, y1  , col);
                set_pixel(x1-i, y1  , col);
                set_pixel(x1  , y1+i, col);
                set_pixel(x1  , y1-i, col);
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

    void print_colors_by_frequency(){
        // Map to store the frequency of each color
        unordered_map<int, int> color_frequency;

        // Traverse the pixels vector and count the occurrences of each color
        for (int x = 0; x < w; x++) for (int y = 0; y < h; y++)
            color_frequency[get_pixel(x, y)]++;

        // Convert the map to a vector of pairs (color, frequency)
        vector<pair<int, int>> frequency_vector(color_frequency.begin(), color_frequency.end());

        // Sort the vector by frequency in descending order
        sort(frequency_vector.begin(), frequency_vector.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
            return b.second < a.second;
        });

        // Print the top 5 most common colors
        cout << "Top 5 most common colors:" << endl;
        for (int i = 0; i < min(5, static_cast<int>(frequency_vector.size())); ++i) {
            int color = frequency_vector[i].first;
            int frequency = frequency_vector[i].second;
            cout << "Color: " << color_to_string(color) << ", Frequency: " << frequency << endl;
        }
    }

    inline float fractional_part(float x) {return x - floor(x);}
    void xiaolin_wu(int x0, int y0, int x1, int y1, int col) {
        bool steep = abs(y1 - y0) > abs(x1 - x0);
        if (steep) {
            std::swap(x0, y0);
            std::swap(x1, y1);
        }
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }

        int dx = x1 - x0;
        int dy = y1 - y0;
        float gradient = dy / static_cast<float>(dx);

        // Compute start and end points
        int xend = round(x0);
        float yend = y0 + gradient * (xend - x0);
        float xgap = 1.0 - fractional_part(x0 + 0.5);
        int xpxl1 = xend;
        int ypxl1 = floor(yend);
        if (steep) {
            set_pixel(ypxl1, xpxl1, col * (1 - fractional_part(yend) * xgap));
            set_pixel(ypxl1 + 1, xpxl1, col * fractional_part(yend) * xgap);
        } else {
            set_pixel(xpxl1, ypxl1, col * (1 - fractional_part(yend) * xgap));
            set_pixel(xpxl1, ypxl1 + 1, col * fractional_part(yend) * xgap);
        }

        // Compute end points
        xend = round(x1);
        yend = y1 + gradient * (xend - x1);
        xgap = fractional_part(x1 + 0.5);
        int xpxl2 = xend;
        int ypxl2 = floor(yend);
        if (steep) {
            set_pixel(ypxl2, xpxl2, col * (1 - fractional_part(yend) * xgap));
            set_pixel(ypxl2 + 1, xpxl2, col * fractional_part(yend) * xgap);
        } else {
            set_pixel(xpxl2, ypxl2, col * (1 - fractional_part(yend) * xgap));
            set_pixel(xpxl2, ypxl2 + 1, col * fractional_part(yend) * xgap);
        }

        // Main loop
        if (steep) {
            for (int x = xpxl1 + 1; x < xpxl2; x++) {
                set_pixel(floor(gradient * (x - x0) + y0), x, col * (1 - fractional_part(gradient * (x - x0) + y0)));
                set_pixel(floor(gradient * (x - x0) + y0) + 1, x, col * fractional_part(gradient * (x - x0) + y0));
            }
        } else {
            for (int x = xpxl1 + 1; x < xpxl2; x++) {
                set_pixel(x, floor(gradient * (x - x0) + y0), col * (1 - fractional_part(gradient * (x - x0) + y0)));
                set_pixel(x, floor(gradient * (x - x0) + y0) + 1, col * fractional_part(gradient * (x - x0) + y0));
            }
        }
    }

    void rounded_rect(int x, int y, int rw, int rh, int r, int col){
        fill_rect(x+r, y, rw-r*2, rh, col);
        fill_rect(x, y+r, rw, rh-r*2, col);
        for(int i = 0; i < 4; i++)
            fill_ellipse(i%2==0 ? (x+r) : (x+w-r), i/2==0 ? (y+r) : (y+rh-r), r, r, col);
    }

    void flood_fill(int x, int y, int color) {
        int targetColor = get_pixel(x, y);

        // Check if the target pixel already has the desired color
        if (targetColor == color)
            return;

        std::stack<std::pair<int, int>> stack;
        stack.push({x, y});

        while (!stack.empty()) {
            auto [curX, curY] = stack.top();
            stack.pop();

            // Check if current pixel is within the image boundaries
            if (curX < 0 || curX >= w || curY < 0 || curY >= h)
                continue;

            // Check if current pixel is the target color
            if (get_pixel(curX, curY) != targetColor)
                continue;

            // Update the pixel color
            set_pixel(curX, curY, color);

            // Push neighboring pixels onto the stack
            stack.push({curX - 1, curY}); // Left
            stack.push({curX + 1, curY}); // Right
            stack.push({curX, curY - 1}); // Up
            stack.push({curX, curY + 1}); // Down
        }
    }

    void filter_greenify_grays() {
        for(int i = 2; i < pixels.size(); i+=4)
            pixels[i] = (pixels[i]*pixels[i])/255;
        for(int i = 1; i < pixels.size(); i+=4)
            pixels[i] = 255-square(255-pixels[i])/255;
        //for(int i = 0; i < pixels.size(); i+=4)
        //    pixels[i] = (pixels[i]*pixels[i])/255;
    }

};

Pixels create_alpha_from_intensities(const vector<vector<int>>& intensities, int negative_intensity) {
    int height = intensities.size();
    int width = (height > 0) ? intensities[0].size() : 0;

    Pixels result(width, height);
    result.fill(OPAQUE_WHITE);

    // Find the minimum and maximum intensity values
    int minIntensity = numeric_limits<int>::max();
    int maxIntensity = numeric_limits<int>::min();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int intensity = intensities[y][x];
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
            int intensity = intensities[y][x];
            int normalizedIntensity = (intensity - minIntensity) * 255 / intensityRange;
            if(intensity < 0) normalizedIntensity = negative_intensity;
            result.set_alpha(x, y, normalizedIntensity);
        }
    }

    return result;
}

Pixels crop(const Pixels& p) {
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
            cropped_pixels.set_pixel(x - min_x, y - min_y, p.get_pixel(x, y));
        }
    }

    return cropped_pixels;
}
