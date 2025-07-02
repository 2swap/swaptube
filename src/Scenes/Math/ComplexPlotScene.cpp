#pragma once

#include "../Common/CoordinateScene.cpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

extern "C" void color_complex_polynomial(
    unsigned int* h_pixels, // to be overwritten with the result
    int w,
    int h,
    const double* h_coefficients_real,
    const double* h_coefficients_imag,
    int degree,
    double lx, double ty,
    double rx, double by
);

class ComplexPlotScene : public CoordinateScene {
public:
    ComplexPlotScene(const double width = 1, const double height = 1) : CoordinateScene(width, height) {
        for(string type : {"coefficient", "root"})
            for(int num = 0; num <= 6; num++)
                for(char ri : {'r', 'i'})
                    state_manager.set(type + to_string(num) + "_" + ri, "0");
        state_manager.set("roots_or_coefficients_control", "0"); // Default to root control
    }

    void state_manager_roots_to_coefficients(){
        vector<complex<double>> coefficients = get_coefficients();
        
        for(int point_index = 0; state_manager.contains("root_r" + to_string(point_index)); point_index++) {
            state_manager.remove_equation("root_r" + to_string(point_index));
            state_manager.remove_equation("root_i" + to_string(point_index));
        }
        for(int i = 0; i < coefficients.size(); i++){
            state_manager.set("coefficient_r" + to_string(i), to_string(coefficients[i].real()));
            state_manager.set("coefficient_i" + to_string(i), to_string(coefficients[i].imag()));
        }
        state_manager.evaluate_all();
    }

    void state_manager_coefficients_to_roots(){
        vector<complex<double>> roots = get_roots();
         
        for(int point_index = 0; state_manager.contains("coefficient_r" + to_string(point_index)); point_index++) {
            state_manager.remove_equation("coefficient_r" + to_string(point_index));
            state_manager.remove_equation("coefficient_i" + to_string(point_index));
        }
        for(int i = 0; i < roots.size(); i++){
            state_manager.set("root_r" + to_string(i), to_string(roots[i].real()));
            state_manager.set("root_i" + to_string(i), to_string(roots[i].imag()));
        }
        state_manager.evaluate_all();
    }

    vector<complex<double>> get_coefficients(){
        if(state["roots_or_coefficients_control"] != 0) {
            vector<complex<double>> coefficients;
            for(int point_index = 0; point_index <= 6; point_index++) {
                string key = "coefficient" + to_string(point_index) + "_";
                coefficients.push_back(complex<double>(state[key + "r"], state[key + "i"]));
            }
            return coefficients;
        }
        
        // The initial polynomial P(z) = 1
        vector<complex<double>> current_poly = {1.0};

        for (const auto& root : get_roots()) {
            // Each new polynomial will have one degree higher than the previous
            vector<complex<double>> new_poly(current_poly.size() + 1);

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
    vector<complex<double>> compute_derivative(const vector<complex<double>>& coefficients) {
        vector<complex<double>> derivative;
        int n = coefficients.size() - 1; // Degree of the polynomial

        for (int i = 1; i <= n; ++i) {
            derivative.push_back(coefficients[i] * complex<double>(i, 0));
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
        if(state["roots_or_coefficients_control"] == 0) {
            for(int point_index = 0; point_index <= 6; point_index++) {
                string key = "root" + to_string(point_index) + "_";
                roots.push_back(complex<double>(state[key + "r"], state[key + "i"]));
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
                cout << "Eigenvalue computation did not converge." << endl;
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

    void draw() override{
        render_root_mode(get_coefficients(), get_roots());
        render_coefficient_mode(get_coefficients());
        CoordinateScene::draw();
    }

    void render_root_mode(const vector<complex<double>>& coefficients, const vector<complex<double>>& roots){
        cout << "1" << endl;
        int w = get_width();
        int h = get_height();

        // Decompose coefficients into separate real and imag arrays for CUDA call
        double h_coefficients_real[7];
        double h_coefficients_imag[7];
        for (int i = 0; i <= 6; ++i) {
            if (i < coefficients.size()) {
                h_coefficients_real[i] = coefficients[i].real();
                h_coefficients_imag[i] = coefficients[i].imag();
            } else {
                h_coefficients_real[i] = 0.0;
                h_coefficients_imag[i] = 0.0;
            }
        }
        cout << "2" << endl;

        /*color_complex_polynomial(
            pix.pixels.data(),
            pix.w,
            pix.h,
            h_coefficients_real,
            h_coefficients_imag,
            6,
            state["left_x"], state["top_y"],
            state["right_x"], state["bottom_y"]
        );*/

        double gm = get_geom_mean_size() / 100;
        cout << "3" << endl;
        for(int i = 0; i < roots.size(); i++){
        cout << "3" << i << endl;
            const complex<int> pixel(point_to_pixel(roots[i]));
        cout << "4" << i << endl;
        cout << pixel.real() << endl;
        cout << pixel.imag() << endl;
        cout << gm << endl;
            pix.fill_ellipse(pixel.real(), pixel.imag(), gm, gm, OPAQUE_WHITE);
        cout << "5" << i << endl;
        }
        cout << "6" << endl;
    }

    void render_coefficient_mode(const vector<complex<double>>& coefficients){
        for(int i = 0; i < coefficients.size(); i++){
            cout << "Aiii" << i << endl;
            double opa = clamp(0,abs(coefficients[i]),1);
            if(opa < 0.01) continue;
            const complex<int> pixel(point_to_pixel(coefficients[i]));
            ScalingParams sp = ScalingParams(get_width() / 6, get_height() / 6);
            Pixels text_pixels = latex_to_pix("x^" + to_string(i), sp);
            pix.overlay(text_pixels, pixel.real() - text_pixels.w / 2, pixel.imag() - text_pixels.h / 2, opa);
            cout << "Biii" << i << endl;
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        for(string type : {"coefficient", "root"})
            for(int num = 0; num <= 6; num++)
                for(char ri : {'r', 'i'})
                    sq.insert(type + to_string(num) + "_" + ri);
        sq.insert("roots_or_coefficients_control");
        return sq;
    }
    void mark_data_unchanged() override { }
    void change_data() override { } // ComplexPlotScene has no DataObjects
    bool check_if_data_changed() const override { return false; } // ComplexPlotScene has no DataObjects
};
