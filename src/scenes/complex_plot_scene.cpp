#pragma once

#include "scene.cpp"
#include <complex>

enum complex_plot_mode{
    ROOTS,
    COEFFICIENTS
};

class ComplexPlotScene : public Scene {
public:
    ComplexPlotScene(const int width, const int height) : Scene(width, height) {}
    ComplexPlotScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    std::pair<int, int> coordinate_to_pixel(std::complex<double> coordinate){
        return std::make_pair(coordinate.real()/pixel_width + w/2., coordinate.imag()/pixel_width + h/2.);
    }

    std::complex<double> pixel_to_coordinate(std::pair<int, int> pixel){
        return std::complex<double>((pixel.first - w/2.) * pixel_width, (pixel.second - h/2.) * pixel_width);
    }

    int complex_to_color(const std::complex<double>& c) {
        float hue = std::arg(c) * 180 / M_PI + 180;  // Convert [-π, π] to [0, 1]
        float saturation = 1.0f;
        float value = (2/M_PI) * atan(std::abs(c));

        int r, g, b;
        hsv2rgb(hue, saturation, value, r, g, b);

        return rgb_to_col(r, g, b) | 0xff000000;
    }

    void update_variables(const std::unordered_map<string, double>& variables) {
        for(int point_index = 0; point_index < roots.size(); point_index++) {
            double r, i;
            set_variable(r, "r" + std::to_string(point_index), variables);
            set_variable(i, "i" + std::to_string(point_index), variables);
            roots[point_index] = std::complex<double>(r, i);
        }
        set_variable(pixel_width, "pixel_width", variables);
    }

    std::complex<double> polynomial(const complex<double>& c){
        std::complex<double> out(1, 0);
        for(std::complex<double> point : roots){
            out *= c - point;
        }
        return out;
    }

    Pixels* query(bool& done_scene) override {
        //if(is_transition) interpolate();
        if(!rendered){
            render_plot();
            rendered = true;
        }
        done_scene = time++>=scene_duration_frames;
        //if(done_scene && is_transition) post_transition();
        return &pix;
    }

    void set_mode(complex_plot_mode cpm){
        mode = cpm;
    }

    void render_plot(){
        if(mode == ROOTS)render_roots();
        if(mode == COEFFICIENTS)render_coefficients();
        pix.add_border(WHITE);
    }

    void add_point(double x, double y){
        roots.push_back(std::complex<double>(x, y));
        rendered = false;
    }

    void set_pixel_width(double w){
        pixel_width = w;
        rendered = false;
    }

    double get_pixel_width(){
        return pixel_width;
    }

    void render_point(std::complex<double> point){
        std::pair<int, int> pixel = coordinate_to_pixel(point);
        pix.fill_ellipse(pixel.first, pixel.second, 5, 5, WHITE);
    }

    void render_roots(){
        pix.fill(BLACK);
        for(int x = 0; x < w; x++){
            for(int y = 0; y < h; y++){
                pix.set_pixel(x, y, complex_to_color(polynomial(pixel_to_coordinate(make_pair(x, y)))));
            }
        }
        for(std::complex<double> point : roots){
            render_point(point);
        }
    }

    void render_coefficients(){
        pix.fill(BLACK);
        determine_coefficients_from_roots();
        for(std::complex<double> point : coefficients){
            render_point(point);
        }
    }

    void determine_coefficients_from_roots() {
        // The initial polynomial P(z) = 1
        std::vector<std::complex<double>> current_poly = {1.0};

        for (const auto& root : roots) {
            // Each new polynomial will have one degree higher than the previous
            std::vector<std::complex<double>> new_poly(current_poly.size() + 1);

            // Multiply the current polynomial by (z - root)
            for (size_t i = 0; i < current_poly.size(); ++i) {
                new_poly[i] += current_poly[i] * (-root);     // Coefficient for (current_poly * -root)
                new_poly[i + 1] += current_poly[i];           // Coefficient for (current_poly * z)
            }

            current_poly = new_poly;
        }

        coefficients = current_poly;
    }

private:
    std::vector<std::complex<double>> roots;
    std::vector<std::complex<double>> coefficients;
    double pixel_width = 0.01;
    complex_plot_mode mode = ROOTS;
};
