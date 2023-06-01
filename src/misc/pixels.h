#pragma once

#include <vector>
#include <librsvg-2.0/librsvg/rsvg.h>
#include "inlines.h"
#include <unistd.h>  // For access() function
#include <fstream>
#include <sstream>

using namespace std;

inline int RED         = 0x880000;
inline int YELLOW      = 0x888800;
inline int EMPTY       = 0x003388;

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
        pixels[spot+3] = 255;
    }

    inline void set_pixel_with_transparency(int x, int y, int col) {
        if(x < 0 || x >= w || y < 0 || y >= h) return;
        // this could go in a loop but unrolled for speeeeed
        int spot = 4*(w*y+x);
        int alpha = geta(col);
        int mergecol = colorlerp(makecol(pixels[spot+2], pixels[spot+1], pixels[spot+0]), col&0xffffff, alpha/255.);
        pixels[spot+0] = getb(mergecol);
        pixels[spot+1] = getg(mergecol);
        pixels[spot+2] = getr(mergecol);
        pixels[spot+3] = 255;
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

    void draw_c4_disk(int stonex, int stoney, int col, bool highlight, char annotation){
        double stonewidth = w/16.;
        int highlightcol = colorlerp(col, 0, .4);
        int textcol = colorlerp(col, 0, .7);
        double px = (stonex-WIDTH/2.+.5)*stonewidth+w/2;
        double py = (-stoney+HEIGHT/2.-.5)*stonewidth+h/2;
        if(highlight) fill_ellipse(px, py, stonewidth*.48, stonewidth*.48, highlightcol);
        fill_ellipse(px, py, stonewidth*.4, stonewidth*.4, col);

        switch (annotation) {
            case '+':
                fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, textcol);  // Draw two rectangles to form a plus sign
                fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, textcol);
                break;
            case '-':
                fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, textcol);  // Draw a rectangle to form a minus sign
                break;
            case '|':
                fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, textcol);  // Draw a rectangle to form a vertical bar
                break;
            case '=':
                fill_rect(px - stonewidth/4 , py - 3*stonewidth/16, stonewidth/2, stonewidth/8, textcol);  // Draw two rectangles to form an equal sign
                fill_rect(px - stonewidth/4 , py + stonewidth/16, stonewidth/2, stonewidth/8, textcol);
                break;
            case 't':
                fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, textcol);  // Draw a rectangle to form a 't'
                fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, textcol);
                break;
            case 'T':
                fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, textcol);  // Draw a rectangle to form a 'T'
                fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, textcol);
                break;
            case ':':
                fill_ellipse(px, py - stonewidth / 8, stonewidth / 12, stonewidth / 12, textcol);  // Draw an ellipse to form a ':'
                fill_ellipse(px, py + stonewidth / 8, stonewidth / 12, stonewidth / 12, textcol);
                break;
            case '0':
                fill_ellipse(px, py, stonewidth / 4, stonewidth / 3, textcol);  // Draw a circle to form a '0'
                fill_ellipse(px, py, stonewidth / 9, stonewidth / 5, col);
                break;
            case '.':
                fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, textcol);  // Draw a circle to form a '0'
                break;
            default:
                break;
        }
    }

    void render_c4_board(Board b){
        int cols[] = {EMPTY, RED, YELLOW};

        // background
        fill(0);
        for(int stonex = 0; stonex < WIDTH; stonex++)
            for(int stoney = 0; stoney < HEIGHT; stoney++)
                draw_c4_disk(stonex, stoney, cols[b.grid[stoney][stonex]], b.highlight[stoney][stonex], b.get_annotation(stonex, stoney));

    }
};

int convolve(const Pixels& a, const Pixels& b, int adx, int ady){
    /*b should be smaller for speed*/
    if(a.w*a.h<b.w*b.h) return convolve(b, a, -adx, -ady);

    double sum = 0;
    for (int x = 0; x < b.w; x++)
        for (int y = 0; y < b.h; y++){
            sum += .3 * a.get_alpha(x-adx, y-ady) * b.get_alpha(x, y);
        }
    return sum;
}

Pixels convolve_map(const Pixels& p1, const Pixels& p2, int& max_x, int& max_y, bool complete){
    int max_conv = 0;
    Pixels ret(p1.w+p2.w, p1.h+p2.h);
    int jump = complete ? 1 : 6;
    for(int x = 0; x < ret.w; x+=jump)
        for(int y = 0; y < ret.h; y+=jump){
            int convolution = convolve(p1, p2, x-p1.w, y-p1.h);
            if(convolution > max_conv){
                max_conv = convolution;
                max_x = x;
                max_y = y;
            }
            ret.set_pixel(x, y, convolution);
        }
    for(int x = max(0, max_x-jump*2); x < min(ret.w, max_x+jump*2); x++)
        for(int y = max(0, max_y-jump*2); y < min(ret.h, max_y+jump*2); y++){
            int convolution = convolve(p1, p2, x-p1.w, y-p1.h);
            if(convolution > max_conv){
                max_conv = convolution;
                max_x = x;
                max_y = y;
            }
            ret.set_pixel(x, y, convolution);
        }
    max_x -= p1.w;
    max_y -= p1.h;
    return ret;
}

Pixels svg_to_pix(const string& svg, int scale_factor) {
    //Open svg and get its dimensions
    RsvgHandle* handle = rsvg_handle_new_from_file(svg.c_str(), NULL);
    if (!handle)
    {
        fprintf(stderr, "Error loading SVG data from file \"%s\"\n", svg.c_str());
        exit(-1);
    }
    RsvgDimensionData dimensionData; 
    rsvg_handle_get_dimensions(handle, &dimensionData);

    int w = dimensionData.width*scale_factor;
    int h = dimensionData.height*scale_factor;
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
    return ret;
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
