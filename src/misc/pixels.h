#pragma once

#include <vector>
#include <librsvg-2.0/librsvg/rsvg.h>
#include "inlines.h"
#include <unistd.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <fstream>
#include <sstream>
#include <stack>
#include <iomanip>
#include <unordered_map>
#include <algorithm>

using namespace std;

inline int BLACK = 0xFF000000;
inline int WHITE = 0xFFFFFFFF;
inline int TRANSPARENT_BLACK = 0x00000000;
inline int TRANSPARENT_WHITE = 0x00FFFFFF;

class Pixels{
public:
    vector<uint8_t> pixels;
    int w;
    int h;
    Pixels() : w(0), h(0), pixels(0){};
    Pixels(int width, int height) : w(width), h(height), pixels(4*width*height){};
    // Copy constructor
    Pixels(const Pixels& other) : w(other.w), h(other.h), pixels(other.pixels) {};

    inline bool out_of_range(int x, int y) const {
        return x < 0 || x >= w || y < 0 || y >= h;
    }

    inline int get_pixel(int x, int y) const {
        if(out_of_range(x, y)) return 0;
        int spot = 4*(w*y+x);
        return makecol(pixels[spot+3], pixels[spot+2], pixels[spot+1], pixels[spot+0]);
    }

    inline int get_alpha(int x, int y) const {
        if(out_of_range(x, y)) return 0;
        return pixels[4*(w*y+x)+3];
    }

    inline void set_pixel(int x, int y, int col) {
        if(out_of_range(x, y)) return;
        // this could go in a loop but unrolled for speeeeed
        int spot = 4*(w*y+x);
        pixels[spot+0] = getb(col);
        pixels[spot+1] = getg(col);
        pixels[spot+2] = getr(col);
        pixels[spot+3] = geta(col);
    }

    inline void set_alpha(int x, int y, int a) {
        if(out_of_range(x, y) || a < 0 || a > 255) return;
        pixels[4*(w*y+x)+3] = a;
    }

    inline void set_pixel_with_transparency(int x, int y, int col) {
        if(out_of_range(x, y)) return;
        // this could go in a loop but unrolled for speeeeed
        int spot = 4*(w*y+x);
        int upper_alpha = geta(col);
        int mergecol = colorlerp(makecol(pixels[spot+2], pixels[spot+1], pixels[spot+0]), col, upper_alpha/255.);
        pixels[spot+0] = getb(mergecol);
        pixels[spot+1] = getg(mergecol);
        pixels[spot+2] = getr(mergecol);
        pixels[spot+3] = 255 - (255-upper_alpha) * (255-get_alpha(x, y)) / 255;
    }

    void print_dimensions(){
        cout << "w: " << w << ", h: " << h << endl;
    }

    void print_to_terminal() const {
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
                int sampleX = x * xStep;
                int sampleY = y * yStep;

                int r = pixels[(sampleX + w * sampleY) * 4];
                int g = pixels[(sampleX + w * sampleY) * 4 + 1];
                int b = pixels[(sampleX + w * sampleY) * 4 + 2];

                // Map the RGB values to ANSI color codes
                int rCode = static_cast<int>((1-cube(1-(r / 256.0))) * 5);
                int gCode = static_cast<int>((1-cube(1-(g / 256.0))) * 5);
                int bCode = static_cast<int>((1-cube(1-(b / 256.0))) * 5);

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

    void add_border(int col){
        for(int x = 0; x < w; x++){
            set_pixel(x, 0, col);
            set_pixel(x, h-1, col);
        }
        for(int y = 0; y < h; y++){
            set_pixel(0, y, col);
            set_pixel(w-1, y, col);
        }
    }

    int bilinear_interpolate(int a, int b, int c, int d, double alpha, double beta) {
        return a * (1 - alpha) * (1 - beta) + b * alpha * (1 - beta) + c * (1 - alpha) * beta + d * alpha * beta;
    }

    void copy_and_scale_integer(Pixels p, int x, int y, int scale, double transparency){
        for(int dx = 0; dx < p.w/scale; dx++)
            for(int dy = 0; dy < p.h/scale; dy++){
                int r = 0;
                int g = 0;
                int b = 0;
                int a = 0;
                for(int scalex = 0; scalex < scale; scalex++)
                    for(int scaley = 0; scaley < scale; scaley++){
                        int spot = 4*(p.w*(dy*scale+scaley)+(dx*scale+scalex));
                        b += p.pixels[spot];
                        g += p.pixels[spot+1];
                        r += p.pixels[spot+2];
                        a += p.pixels[spot+3];
                    }
                r /= square(scale);
                g /= square(scale);
                b /= square(scale);
                a /= square(scale);

                int col = (int(a*transparency)<<24)+(r<<16)+(g<<8)+b;
                set_pixel_with_transparency(x+dx, y+dy, col);
            }
    }

    void copy_and_scale_bilinear(Pixels p, int x, int y, double scale) {
        int newWidth = static_cast<int>(p.w * scale);
        int newHeight = static_cast<int>(p.h * scale);

        for (int dx = 0; dx < newWidth; dx++) {
            for (int dy = 0; dy < newHeight; dy++) {
                // Bilinear interpolation
                double gx = (dx / scale);
                double gy = (dy / scale);

                int gxi = static_cast<int>(gx);
                int gyi = static_cast<int>(gy);

                int c00 = p.get_pixel(gxi, gyi);
                int c10 = p.get_pixel(gxi + 1, gyi);
                int c01 = p.get_pixel(gxi, gyi + 1);
                int c11 = p.get_pixel(gxi + 1, gyi + 1);

                double alpha = gx - gxi;
                double beta = gy - gyi;

                int r = bilinear_interpolate(c00 >> 16 & 0xFF, c10 >> 16 & 0xFF, c01 >> 16 & 0xFF, c11 >> 16 & 0xFF, alpha, beta);
                int g = bilinear_interpolate(c00 >> 8 & 0xFF, c10 >> 8 & 0xFF, c01 >> 8 & 0xFF, c11 >> 8 & 0xFF, alpha, beta);
                int b = bilinear_interpolate(c00 & 0xFF, c10 & 0xFF, c01 & 0xFF, c11 & 0xFF, alpha, beta);
                int a = bilinear_interpolate(c00 >> 24 & 0xFF, c10 >> 24 & 0xFF, c01 >> 24 & 0xFF, c11 >> 24 & 0xFF, alpha, beta);

                int col = (a << 24) + (r << 16) + (g << 8) + b;
                set_pixel_with_transparency(x + dx, y + dy, col);
            }
        }
    }

    void copy(Pixels p, int dx, int dy, double transparency){
        for(int x = 0; x < p.w; x++){
            int xpdx = x+dx;
            for(int y = 0; y < p.h; y++){
                int col = p.get_pixel(x, y);
                col = (int(geta(col)*transparency)<<24) + (col & TRANSPARENT_WHITE);
                set_pixel_with_transparency(x+dx, y+dy, col);
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
        for(int i = 0; i < pixels.size(); i+=4){
            if(pixels[i] != pixels[i+1] || pixels[i+2] != pixels[i+1]) continue;
            pixels[i+3] = pixels[i];
            pixels[i] = 255;
            pixels[i+1] = 255;
            pixels[i+2] = 255;
        }
    }

    void fill_rect(int x, int y, int rw, int rh, int col){
        for(int dx = 0; dx < rw; dx++)
            for(int dy = 0; dy < rh; dy++)
                set_pixel(x+dx, y+dy, col);
    }

    void fill_ellipse(double x, double y, double rw, double rh, int col){
        for(double dx = -rw+1; dx < rw; dx++)
            for(double dy = -rh+1; dy < rh; dy++)
                if(square(dx/rw)+square(dy/rh) < 1)
                    set_pixel_with_transparency(x+dx, y+dy, col);
    }

    void fill(int col){
        fill_rect(0, 0, w, h, col);
    }

    void fill_alpha(int a){
        for(int dx = 0; dx < w; dx++)
            for(int dy = 0; dy < h; dy++){
                pixels[4*(dx+dy*w)+3] = a;
            }
    }

    void mult_alpha(double m){
        for(int dx = 0; dx < w; dx++)
            for(int dy = 0; dy < h; dy++){
                pixels[4*(dx+dy*w)+3] *= m;
            }
    }

    void mult_color(double m){
        for(int dx = 0; dx < w; dx++)
            for(int dy = 0; dy < h; dy++){
                pixels[4*(dx+dy*w)+0] *= m;
                pixels[4*(dx+dy*w)+1] *= m;
                pixels[4*(dx+dy*w)+2] *= m;
            }
    }

    void recolor(int col){
        int r = getr(col);
        int g = getg(col);
        int b = getb(col);
        for(int dx = 0; dx < w; dx++)
            for(int dy = 0; dy < h; dy++){
                pixels[4*(dx+dy*w)+2] = r;
                pixels[4*(dx+dy*w)+1] = g;
                pixels[4*(dx+dy*w)+0] = b;
            }
    }

    void bresenham(int x1, int y1, int x2, int y2, int col, int thickness) {
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);

        int sx = (x1 < x2) ? 1 : -1; // Direction of drawing on x axis
        int sy = (y1 < y2) ? 1 : -1; // Direction of drawing on y axis

        int err = dx - dy;

        while(true) {
            set_pixel(x1, y1, col);
            for(int i = 1; i < thickness; i++){
                set_pixel(x1+1, y1  , col);
                set_pixel(x1-1, y1  , col);
                set_pixel(x1  , y1+1, col);
                set_pixel(x1  , y1-1, col);
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
            set_pixel_with_transparency(ypxl1, xpxl1, col * (1 - fractional_part(yend) * xgap));
            set_pixel_with_transparency(ypxl1 + 1, xpxl1, col * fractional_part(yend) * xgap);
        } else {
            set_pixel_with_transparency(xpxl1, ypxl1, col * (1 - fractional_part(yend) * xgap));
            set_pixel_with_transparency(xpxl1, ypxl1 + 1, col * fractional_part(yend) * xgap);
        }

        // Compute end points
        xend = round(x1);
        yend = y1 + gradient * (xend - x1);
        xgap = fractional_part(x1 + 0.5);
        int xpxl2 = xend;
        int ypxl2 = floor(yend);
        if (steep) {
            set_pixel_with_transparency(ypxl2, xpxl2, col * (1 - fractional_part(yend) * xgap));
            set_pixel_with_transparency(ypxl2 + 1, xpxl2, col * fractional_part(yend) * xgap);
        } else {
            set_pixel_with_transparency(xpxl2, ypxl2, col * (1 - fractional_part(yend) * xgap));
            set_pixel_with_transparency(xpxl2, ypxl2 + 1, col * fractional_part(yend) * xgap);
        }

        // Main loop
        if (steep) {
            for (int x = xpxl1 + 1; x < xpxl2; x++) {
                set_pixel_with_transparency(floor(gradient * (x - x0) + y0), x, col * (1 - fractional_part(gradient * (x - x0) + y0)));
                set_pixel_with_transparency(floor(gradient * (x - x0) + y0) + 1, x, col * fractional_part(gradient * (x - x0) + y0));
            }
        } else {
            for (int x = xpxl1 + 1; x < xpxl2; x++) {
                set_pixel_with_transparency(x, floor(gradient * (x - x0) + y0), col * (1 - fractional_part(gradient * (x - x0) + y0)));
                set_pixel_with_transparency(x, floor(gradient * (x - x0) + y0) + 1, col * fractional_part(gradient * (x - x0) + y0));
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
    result.fill(WHITE);

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

Pixels svg_to_pix(const string& svg, int scale_factor) {
    //Open svg and get its dimensions
    RsvgHandle* handle = rsvg_handle_new_from_file(svg.c_str(), NULL);
    if (!handle)
    {
        fprintf(stderr, "Error loading SVG data from file \"%s\"\n", svg.c_str());
        exit(-1);
    }
    gdouble dim_width = 0;
    gdouble dim_height = 0; 
    rsvg_handle_get_intrinsic_size_in_pixels(handle, &dim_width, &dim_height);

    int w = dim_width*scale_factor;
    int h = dim_height*scale_factor;
    Pixels ret(w, h);

    //Render it
    cairo_surface_t* surface = cairo_image_surface_create_for_data(&(ret.pixels[0]), CAIRO_FORMAT_ARGB32, w, h, w * 4);
    cairo_t* cr = cairo_create(surface);
    cairo_scale(cr, scale_factor, scale_factor);
    rsvg_handle_render_cairo(handle, cr);
    g_object_unref(handle);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    ret.grayscale_to_alpha();
    return crop(ret);
}

// Create an unordered_map to store the cached results
std::unordered_map<std::string, Pixels> latex_cache;

Pixels eqn_to_pix(const string& eqn, int scale_factor){
    // Check if the result is already in the cache
    auto it = latex_cache.find(eqn);
    if (it != latex_cache.end()) {
        return it->second; // Return the cached Pixels object
    }

    hash<string> hasher;
    string name = "/home/swap/CS/swaptube/out/latex/" + to_string(hasher(eqn)) + ".svg";

    if (access(name.c_str(), F_OK) != -1) {
        // File already exists, no need to generate LaTeX
        Pixels pixels = svg_to_pix(name, scale_factor);
        latex_cache[eqn] = pixels; // Cache the result before returning
        return pixels;
    }

    string command = "cd ../../MicroTeX-master/build/ && ./LaTeX -headless -foreground=#ffffffff \"-input=" + eqn + "\" -output=" + name + " >/dev/null 2>&1";
    int result = system(command.c_str());
    
    if (result == 0) {
        // System call successful, return the generated SVG
        Pixels pixels = svg_to_pix(name, scale_factor);
        latex_cache[eqn] = pixels; // Cache the result before returning
        return pixels;
    } else {
        // System call failed, handle the error
        throw runtime_error("Failed to generate LaTeX.");
    }
}
