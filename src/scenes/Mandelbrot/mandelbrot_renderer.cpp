#pragma once

#include "mandelbrot.cpp"
#include "palette.cpp"
#include <queue>

class MandelbrotRenderer {
    Mandelbrot m;
    Palette palette;
    int time = 0;
    public:
        enum WhichParameterization{
            Z,
            X,
            C
        };
        Complex z0 = Complex(0, 0);
        Complex exponent = Complex(0, 0);
        Complex c = Complex(0, 0);

        WhichParameterization which;

        Complex center = Complex(0, 0);
        Complex zoom_multiplier = Complex(0.95, 0.01);
        Complex current_zoom = Complex(.01, 0);
        double breath = 0;

        MandelbrotRenderer() : z0(0.0, 0.0), exponent(0.0, 0.0), c(0.0, 0.0), which(C), center(0.0, 0.0), current_zoom(0.0, 0.0), zoom_multiplier(0.0, 0.0) {}

        MandelbrotRenderer(Complex _z0, Complex _x, Complex _c, WhichParameterization wp, Complex cnt, Complex cz, Complex zm){
            center = cnt;
            current_zoom = cz;
            zoom_multiplier = zm;
            z0 = _z0;
            exponent = _x;
            c = _c;
            which = wp;
        }

        void depths_to_points(vector<vector<int>> depths, Pixels& p){
            //change depths to colors
            bool use_cyclic_palette = false;
            if (use_cyclic_palette) {
                for(int dx = 0; dx < p.w; dx++) {
                    for(int dy = 0; dy < p.h; dy++) {
                        int depth = depths[dy][dx];
                        int col = depth == -1 ? 0 : palette.prompt(depth+time*breath);
                        p.set_pixel(dx, dy, col);
                    }
                }
            }
            else {
                p.fill(BLACK);
                p.copy(create_alpha_from_intensities(depths, 255), 0, 0, 1);
                p.filter_greenify_grays();
            }
        }

        void edge_detect_render(Pixels& p) {
            Complex screen_center = Complex(p.w*0.5, p.h*0.5);
            p.fill(0x00123456);
            int queries = 0;

            vector<vector<int>> depths(p.h, vector<int>(p.w, -2));

            // Create a queue to store the pixels to be queried
            queue<pair<int, int>> queryQueue;

            // Iterate around the border and add edge pixels to the queue
            for (int i = 0; i < p.h; i++) {
                queryQueue.push({0, i});
                queryQueue.push({p.w-1, i});
            }
            for (int i = 0; i < p.w; i++) {
                queryQueue.push({i, 0});
                queryQueue.push({i, p.h-1});
            }

            // Process the queue
            while (!queryQueue.empty()) {
                auto [x, y] = queryQueue.front();
                queryQueue.pop();
                if(p.out_of_range(x, y)) continue;
                if(depths[y][x] != -2) continue;

                // Query the pixel
                Complex point = (Complex(x, y)-screen_center) * current_zoom + center;
                int depth = 0;
                if (which == Z)
                    depth = m.get_depth_complex(point, exponent, c);
                if (which == X)
                    depth = m.get_depth_complex(z0, point, c);
                if (which == C)
                    depth = m.get_depth_complex(z0, exponent, point);
                queries++;
                depths[y][x] = depth;

                // Check against neighbors
                if (x > 0) {
                    int left_neighbor = depths[y][x-1];
                    if (left_neighbor != -2 && left_neighbor != depth) {
                        for (int dy = -1; dy <= 1; dy++) {
                            queryQueue.push({x - 1, y + dy});
                            queryQueue.push({x    , y + dy});
                        }
                    }
                }
                if (x < p.w - 1){
                    int right_neighbor = depths[y][x+1];
                    if (right_neighbor != -2 && right_neighbor != depth) {
                        for (int dy = -1; dy <= 1; dy++) {
                            queryQueue.push({x + 1, y + dy});
                            queryQueue.push({x    , y + dy});
                        }
                    }
                }
                if (y > 0) {
                    int top_neighbor = depths[y-1][x];
                    if(top_neighbor != -2 && top_neighbor != depth) {
                        for (int dx = -1; dx <= 1; dx++) {
                            queryQueue.push({x + dx, y - 1});
                            queryQueue.push({x + dx, y    });
                        }
                    }
                }
                if (y < p.h - 1){
                    int bottom_neighbor = depths[y+1][x];
                    if (bottom_neighbor != -2 && bottom_neighbor != depth) {
                        for (int dx = -1; dx <= 1; dx++) {
                            queryQueue.push({x + dx, y + 1});
                            queryQueue.push({x + dx, y    });
                        }
                    }
                }
            }
            depths_to_points(depths, p);
            cout << queries << endl;
            current_zoom = current_zoom * zoom_multiplier;
            time++;

            // Perform flood fill to fill the remaining transparent locations
            for (int y = 0; y < p.h; y++) {
                for (int x = 0; x < p.w; x++) {
                    if (depths[y][x] == -2) {
                        int left_depth = (x > 0) ? depths[y][x - 1] : -2;
                        int right_depth = (x < p.w - 1) ? depths[y][x + 1] : -2;
                        int top_depth = (y > 0) ? depths[y - 1][x] : -2;
                        int bottom_depth = (y < p.h - 1) ? depths[y + 1][x] : -2;

                        int color = 0;

                        if (left_depth != -2)
                            color = p.get_pixel(x - 1, y);
                        else if (right_depth != -2)
                            color = p.get_pixel(x + 1, y);
                        else if (top_depth != -2)
                            color = p.get_pixel(x, y - 1);
                        else if (bottom_depth != -2)
                            color = p.get_pixel(x, y + 1);
                        else continue;

                        p.flood_fill(x, y, color);
                    }
                }
            }
        }

};
