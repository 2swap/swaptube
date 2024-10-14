#pragma once

#include "scene.cpp"
#include "Mandelbrot/mandelbrot.cpp"
#include "Mandelbrot/palette.cpp"
#include <queue>
#include <variant>

class MandelbrotScene : public Scene {
public:
    MandelbrotScene(const double width = 1, const double height = 1) : Scene(width, height) {
        current_zoom = Complex(contents["current_zoom"]["real"].get<double>(), contents["current_zoom"]["imag"].get<double>());
        add_audio(contents);
    }
    void query(Pixels*& p) override;
    void update_variables(const unordered_map<string, double>& _variables) {
        variables = _variables;
        zoom_multiplier = Complex(get_or_variable(contents["zoom_multiplier"]["real"]), get_or_variable(contents["zoom_multiplier"]["imag"]));
        z0              = Complex(get_or_variable(contents["z"]["real"]              ), get_or_variable(contents["z"]["imag"]              ));
        exponent        = Complex(get_or_variable(contents["x"]["real"]              ), get_or_variable(contents["x"]["imag"]              ));
        c               = Complex(get_or_variable(contents["c"]["real"]              ), get_or_variable(contents["c"]["imag"]              ));

        string paramValue = contents["WhichParameterization"].get<string>();
        which = C;
        if (paramValue == "Z") {
            which = Z;
        } else if (paramValue == "X") {
            which = X;
        }
    }

private:
    Mandelbrot m;
    Palette palette;
    enum WhichParameterization{
        Z,
        X,
        C
    };
    Complex z0;
    Complex exponent;
    Complex c;

    WhichParameterization which;

    Complex zoom_multiplier;
    Complex current_zoom;
    double breath;
    int depth = 100;
    unordered_map<string, double> variables;

    double get_or_variable(variant<string, double> either){
        return holds_alternative<string>(either) ? variables[get<string>(either)] : get<double>(either);
    }
    void depths_to_points(vector<vector<int>> depths, Pixels& p);
    int render_for_parameterization(Complex point);
    void render_mandelbrot_points(Pixels& p);
    void edge_detect_render(Pixels& p);
    void render(Pixels& p);
};

void MandelbrotScene::depths_to_points(vector<vector<int>> depths, Pixels& p){
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
        p.copy(create_alpha_from_intensities(depths, 255), 0, 0);
        p.filter_greenify_grays();
    }
}

int MandelbrotScene::render_for_parameterization(Complex point) {
    if (which == Z)
        return m.get_depth_complex(z0 + point, exponent        , c        , depth);
    if (which == X)
        return m.get_depth_complex(z0        , exponent + point, c        , depth);
    if (which == C)
        return m.get_depth_complex(z0        , exponent        , c + point, depth);
    return -1000;
}

void MandelbrotScene::render_mandelbrot_points(Pixels& p){
    cout << "Rendering by point method... ";
    Complex screen_center = Complex(p.w*0.5, p.h*0.5);
    vector<vector<int>> depths(p.h, vector<int>(p.w, -2));
    int queries = 0;
    for(int jump = 8; jump >= 1; jump/=2){
        for(int dx = 0; dx < p.w; dx+=jump)
            for(int dy = 0; dy < p.h; dy+=jump) {
                if(depths[dy][dx] != -2) continue;
                Complex point = (Complex(dx, dy)-screen_center) * current_zoom;
                depths[dy][dx] = render_for_parameterization(point);
                queries++;
            }
        if(jump != 1){
            for(int dx = jump*2; dx < p.w-jump*2; dx+=jump)
                for(int dy = jump*2; dy < p.h-jump*2; dy+=jump){
                    int depth = depths[dy][dx];
                    bool uniform = true;
                    for(int x = dx-jump*2; x <= dx+jump*2; x+=jump)
                        for(int y = dy-jump*2; y <= dy+jump*2; y+=jump)
                            if(depths[y][x]!=depth)
                                uniform = false;
                    if(uniform)
                        for(int x = dx-jump; x <= dx+jump; x++)
                            for(int y = dy-jump; y <= dy+jump; y++)
                                depths[y][x] = depth;
                }
        }
    }
    cout << queries << " queries" << endl;
    depths_to_points(depths, p);
    current_zoom = current_zoom * zoom_multiplier;
}

void MandelbrotScene::edge_detect_render(Pixels& p) {
    cout << "Rendering by edge detection method... ";
    Complex screen_center = Complex(p.w*0.5, p.h*0.5);
    p.fill(0);
    int queries = 0;

    vector<vector<int>> depths(p.h, vector<int>(p.w, -2));

    // Create a queue to store the pixels to be queried
    queue<pair<int, int>> queryQueue;

    // Iterate around the border and add edge pixels to the queue
    for (int i = 0; i < p.h; i++) {
        queryQueue.push({0, i});
        queryQueue.push({p.w/3, i});
        queryQueue.push({p.w*2/3, i});
        queryQueue.push({p.w-1, i});
    }
    for (int i = 0; i < p.w; i++) {
        queryQueue.push({i, 0});
        queryQueue.push({i, p.h/3});
        queryQueue.push({i, p.h*2/3});
        queryQueue.push({i, p.h-1});
    }

    // Process the queue
    while (!queryQueue.empty()) {
        auto [x, y] = queryQueue.front();
        queryQueue.pop();
        if(p.out_of_range(x, y)) continue;
        if(depths[y][x] != -2) continue;

        // Query the pixel
        Complex point = (Complex(x, y)-screen_center) * current_zoom;
        int depth = render_for_parameterization(point);
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
    cout << queries << " queries" << endl;
    current_zoom = current_zoom * zoom_multiplier;

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
    
void MandelbrotScene::render(Pixels& p){
    if(which == C)
        edge_detect_render(p);
    if(which == Z)
        render_mandelbrot_points(p);
    if(which == X)
        render_mandelbrot_points(p);
    p.fill_ellipse(p.w/2, p.h/2, 5, 5, 0x44ff0000);
}

const Pixels& MandelbrotScene::query(bool& done_scene) {
    double duration_frames = contents["duration_seconds"].get<int>() * VIDEO_FRAMERATE;

    render(pix);
    
    return pix;
}
