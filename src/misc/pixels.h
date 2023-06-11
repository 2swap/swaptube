#pragma once

#include <vector>
#include <librsvg-2.0/librsvg/rsvg.h>
#include "inlines.h"
#include <unistd.h>  // For access() function
#include <fstream>
#include <sstream>

using namespace std;

inline int BLACK = 0xFF000000;
inline int WHITE = 0xFFFFFFFF;

class Pixels{
public:
    vector<uint8_t> pixels;
    int w;
    int h;
    Pixels(int width, int height) : w(width), h(height), pixels(4*width*height){};

    inline int get_pixel(int x, int y) const {
        if(x < 0 || x >= w || y < 0 || y >= h) return 0;
        int spot = 4*(w*y+x);
        return makecol(pixels[spot+3], pixels[spot+2], pixels[spot+1], pixels[spot+0]);
    }

    inline int get_alpha(int x, int y) const {
        if(x < 0 || x >= w || y < 0 || y >= h) return 0;
        return pixels[4*(w*y+x)+3];
    }

    inline void set_pixel(int x, int y, int col) {
        if(x < 0 || x >= w || y < 0 || y >= h) return;
        // this could go in a loop but unrolled for speeeeed
        int spot = 4*(w*y+x);
        pixels[spot+0] = getb(col);
        pixels[spot+1] = getg(col);
        pixels[spot+2] = getr(col);
        pixels[spot+3] = geta(col);
    }

    inline void set_pixel_with_transparency(int x, int y, int col) {
        if(x < 0 || x >= w || y < 0 || y >= h) return;
        // this could go in a loop but unrolled for speeeeed
        int spot = 4*(w*y+x);
        int lower_alpha = get_alpha(x, y);
        int upper_alpha = geta(col);
        int mergecol = colorlerp(makecol(pixels[spot+2], pixels[spot+1], pixels[spot+0]), col, upper_alpha/255.);
        pixels[spot+0] = getb(mergecol);
        pixels[spot+1] = getg(mergecol);
        pixels[spot+2] = getr(mergecol);
        pixels[spot+3] = geta(mergecol);
    }

    inline void print_dimensions(){
        cout << "w: " << w << ", h: " << h << endl;
    }

    void add_border(){
        for(int x = 0; x < w; x++){
            set_pixel(x, 0, 0xffffff);
            set_pixel(x, h-1, 0xffffff);
        }
        for(int y = 0; y < h; y++){
            set_pixel(0, y, 0xffffff);
            set_pixel(w-1, y, 0xffffff);
        }
    }

    Pixels scale(int scale){
        Pixels scaled(w/scale, h/scale);
        scaled.copy(*this, 0, 0, scale, 1);
        return scaled;
    }

    void copy(Pixels p, int x, int y, int scale, double transparency){
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

    void fill_ellipse(int x, int y, int rw, int rh, int col){
        for(double dx = -rw+1; dx < rw; dx++)
            for(double dy = -rh+1; dy < rh; dy++)
                if(square(dx/rw)+square(dy/rh) < 1)
                    set_pixel(x+dx, y+dy, col);
    }

    void fill(int col){
        fill_rect(0, 0, w, h, col);
    }

    void rounded_rect(int x, int y, int rw, int rh, int r, int col){
        fill_rect(x+r, y, rw-r*2, rh, col);
        fill_rect(x, y+r, rw, rh-r*2, col);
        for(int i = 0; i < 4; i++)
            fill_ellipse(i%2==0 ? (x+r) : (x+w-r), i/2==0 ? (y+r) : (y+rh-r), r, r, col);
    }
};

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
    rsvg_handle_close(handle, nullptr);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    ret.grayscale_to_alpha();
    return crop(ret);
}

Pixels eqn_to_pix(const string& eqn, int scale_factor){
    hash<string> hasher;
    string name = "/home/swap/CS/moviemaker-cpp/out/latex/" + to_string(hasher(eqn)) + ".svg";

    if (access(name.c_str(), F_OK) != -1) {
        // File already exists, no need to generate LaTeX
        return svg_to_pix(name, scale_factor);
    }

    string command = "cd /home/swap/CS/MicroTeX-master/build/ && ./LaTeX -headless -foreground=#ffffffff \"-input=" + eqn + "\" -output=" + name + " >/dev/null 2>&1";
    int result = system(command.c_str());
    
    if (result == 0) {
        // System call successful, return the generated SVG
        return svg_to_pix(name, scale_factor);
    } else {
        // System call failed, handle the error
        throw runtime_error("Failed to generate LaTeX.");
    }
}
