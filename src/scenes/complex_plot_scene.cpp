#pragma once

#include "scene.cpp"
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

enum complex_plot_mode{
    ROOTS,
    COEFFICIENTS
};

using namespace Eigen;

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

    std::complex<double> evaluate_polynomial_given_coefficients(const std::vector<std::complex<double>>& coefficients, const std::complex<double>& point) {
        std::complex<double> result = 0.0;
        std::complex<double> power_of_point = 1.0;
        for (const auto& coefficient : coefficients) {
            result += coefficient * power_of_point;
            power_of_point *= point;
        }
        return result;
    }
    std::complex<double> polynomial(const complex<double>& c, const vector<complex<double>>& roots){
        std::complex<double> out(1, 0);
        for(std::complex<double> point : roots){
            out *= c - point;
        }
        return out;
    }

    void query(bool& done_scene, Pixels*& p) override {
        render_plot();
        done_scene = time++>=scene_duration_frames;
        p = &pix;
    }

    void set_mode(complex_plot_mode cpm){
        mode = cpm;
    }

    void dag_roots_to_coefficients(){
        vector<complex<double>> coefficients = get_coefficients();
        
        for(int point_index = 0; dag.contains("root_r" + to_string(point_index)); point_index++) {
            dag.remove_equation("root_r" + to_string(point_index));
            dag.remove_equation("root_i" + to_string(point_index));
        }
        for(int i = 0; i < coefficients.size(); i++){
            dag.add_equation("coefficient_r" + to_string(i), to_string(coefficients[i].real()));
            dag.add_equation("coefficient_i" + to_string(i), to_string(coefficients[i].imag()));
        }
        dag.evaluate_all();
    }

    void dag_coefficients_to_roots(){
        vector<complex<double>> roots = get_roots();
         
        for(int point_index = 0; dag.contains("coefficient_r" + to_string(point_index)); point_index++) {
            dag.remove_equation("coefficient_r" + to_string(point_index));
            dag.remove_equation("coefficient_i" + to_string(point_index));
        }
        for(int i = 0; i < roots.size(); i++){
            dag.add_equation("root_r" + to_string(i), to_string(roots[i].real()));
            dag.add_equation("root_i" + to_string(i), to_string(roots[i].imag()));
        }
        dag.evaluate_all();
    }

    vector<complex<double>> get_coefficients(){
        if(dag.contains("coefficient_r0")){
            vector<complex<double>> coefficients;
            for(int point_index = 0; dag.contains("coefficient_r" + to_string(point_index)); point_index++) {
                std::complex<double> coeff(dag["coefficient_r" + to_string(point_index)],
                                           dag["coefficient_i" + to_string(point_index)]);
                coefficients.push_back(coeff);
            }
            return coefficients;
        }
        
        // The initial polynomial P(z) = 1
        std::vector<std::complex<double>> current_poly = {1.0};

        for (const auto& root : get_roots()) {
            // Each new polynomial will have one degree higher than the previous
            std::vector<std::complex<double>> new_poly(current_poly.size() + 1);

            // Multiply the current polynomial by (z - root)
            for (size_t i = 0; i < current_poly.size(); ++i) {
                new_poly[i] += current_poly[i] * (-root);     // Coefficient for (current_poly * -root)
                new_poly[i + 1] += current_poly[i];           // Coefficient for (current_poly * z)
            }

            current_poly = new_poly;
        }
        return current_poly;
    }

    // Function to compute the derivative of a polynomial given its coefficients
    std::vector<std::complex<double>> compute_derivative(const std::vector<std::complex<double>>& coefficients) {
        std::vector<std::complex<double>> derivative;
        int n = coefficients.size() - 1; // Degree of the polynomial

        for (int i = 1; i <= n; ++i) {
            derivative.push_back(coefficients[i] * std::complex<double>(i, 0));
        }

        return derivative;
    }

    vector<complex<double>> deflate_polynomial(const vector<complex<double>>& coefficients, const complex<double>& root) {
        int n = coefficients.size() - 1;
        vector<complex<double>> new_coefficients(n);
        new_coefficients[0] = coefficients[0];
        for (int i = 1; i < n; ++i) {
            new_coefficients[i] = coefficients[i] + root * new_coefficients[i - 1];
        }
        return new_coefficients;
    }

    vector<complex<double>> get_roots(){
        vector<complex<double>> roots;
        if(dag.contains("root_r0")){
            for(int point_index = 0; dag.contains("root_r" + to_string(point_index)); point_index++) {
                std::complex<double> root(dag["root_r" + to_string(point_index)], dag["root_i" + to_string(point_index)]);
                roots.push_back(root);
            }
        } else {
            vector<complex<double>> coefficients = get_coefficients();
            int n = coefficients.size() - 1;

            // Create the companion matrix
            MatrixXcd companion_matrix = MatrixXcd::Zero(n, n);
            for (int i = 0; i < n; ++i) {
                companion_matrix(i, n - 1) = -coefficients[i] / coefficients[n];
                if (i < n - 1) {
                    companion_matrix(i + 1, i) = complex<double>(1, 0);
                }
            }

            // Compute the eigenvalues (roots)
            ComplexEigenSolver<MatrixXcd> solver(companion_matrix);
            if (solver.info() != Success) {
                cerr << "Eigenvalue computation did not converge." << endl;
                return roots;
            }
            VectorXcd eigenvalues = solver.eigenvalues();

            // Store the roots
            roots.reserve(n);
            for (int i = 0; i < n; ++i) {
                roots.push_back(eigenvalues[i]);
            }
        }
        return roots;
    }

    void render_plot(){
        if(mode == ROOTS)render_root_mode(get_coefficients(), get_roots());
        if(mode == COEFFICIENTS)render_coefficient_mode(get_coefficients());
        render_axes();
    }

    void set_pixel_width(double w){
        pixel_width = w;
    }

    double get_pixel_width(){
        return pixel_width;
    }

    void render_point(std::complex<double> point){
        std::pair<int, int> pixel = coordinate_to_pixel(point);
        pix.fill_ellipse(pixel.first, pixel.second, 5, 5, WHITE);
    }

    void render_axes(){
        std::pair<int, int> i_pos = coordinate_to_pixel(std::complex<double>(0,10));
        std::pair<int, int> i_neg = coordinate_to_pixel(std::complex<double>(0,-10));
        std::pair<int, int> r_pos = coordinate_to_pixel(std::complex<double>(10,0));
        std::pair<int, int> r_neg = coordinate_to_pixel(std::complex<double>(-10,0));
        pix.bresenham(i_pos.first, i_pos.second, i_neg.first, i_neg.second, 0xff222222, 1);
        pix.bresenham(r_pos.first, r_pos.second, r_neg.first, r_neg.second, 0xff222222, 1);
        for(int i = -9; i < 10; i++){
            std::pair<int, int> i_pos = coordinate_to_pixel(std::complex<double>(i,.1));
            std::pair<int, int> i_neg = coordinate_to_pixel(std::complex<double>(i,-.1));
            std::pair<int, int> r_pos = coordinate_to_pixel(std::complex<double>(.1,i));
            std::pair<int, int> r_neg = coordinate_to_pixel(std::complex<double>(-.1,i));
            pix.bresenham(i_pos.first, i_pos.second, i_neg.first, i_neg.second, 0xff222222, 1);
            pix.bresenham(r_pos.first, r_pos.second, r_neg.first, r_neg.second, 0xff222222, 1);
        }
    }

    void render_root_mode(const vector<complex<double>>& coefficients, const vector<complex<double>>& roots){
        pix.fill(BLACK);
        for(int x = 0; x < w; x++){
            for(int y = 0; y < h; y++){
                pix.set_pixel(x, y, complex_to_color(evaluate_polynomial_given_coefficients(coefficients,pixel_to_coordinate(make_pair(x, y)))));
            }
        }
        for(std::complex<double> point : roots){
            render_point(point);
        }
    }

    void render_coefficient_mode(const vector<complex<double>>& coefficients){
        pix.fill(BLACK);
        for(std::complex<double> point : coefficients){
            render_point(point);
        }
    }

    void determine_coefficients_from_roots() {
    }

private:
    double pixel_width = 0.01;
    complex_plot_mode mode = ROOTS;
};
