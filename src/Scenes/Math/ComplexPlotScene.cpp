#pragma once

#include "../Common/CoordinateScene.cpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

extern "C" void color_complex_polynomial(
    unsigned int* h_pixels, // to be overwritten with the result
    int w,
    int h,
    const float* h_coefficients_real,
    const float* h_coefficients_imag,
    int degree,
    float lx, float ty,
    float rx, float by,
    float ab_dilation,
    float dot_radius
);

class ComplexPlotScene : public CoordinateScene {
public:
    int degree;
    ComplexPlotScene(const int d, const float width = 1, const float height = 1) : CoordinateScene(width, height), degree(d){
        complex_plane = true;
        for(string type : {"coefficient", "root"})
            for(int num = 0; num < degree; num++) {
                state_manager.set(type + to_string(num) + "_opacity", "1");
                for(char ri : {'r', 'i'})
                    state_manager.set(type + to_string(num) + "_" + ri, "0");
            }
        state_manager.set("roots_or_coefficients_control", "0"); // Default to root control
        state_manager.set("ab_dilation", ".5"); // basically saturation
        state_manager.set("coefficients_opacity", "1");
        state_manager.set("roots_opacity", "0");
        state_manager.set("hide_zero_coefficients", "0");
        state_manager.set("dot_radius", "3");
    }

    void state_manager_roots_to_coefficients(){
        update_state();

        // If we are already in coefficients mode, do nothing
        if(state["roots_or_coefficients_control"] != 0) return;

        vector<complex<float>> coefficients = get_coefficients();

        int i = 0;
        for(complex<float> c : coefficients){
            state_manager.set("coefficient" + to_string(i) + "_r", to_string(c.real()));
            state_manager.set("coefficient" + to_string(i) + "_i", to_string(c.imag()));
            i++;
        }
        state_manager.set({{"roots_or_coefficients_control", "1"}});
    }

    void state_manager_coefficients_to_roots(){
        update_state();

        // If we are already in roots mode, do nothing
        if(state["roots_or_coefficients_control"] != 1) return;

        vector<complex<float>> roots = get_roots();
         
        int i = 0;
        for(complex<float> r : roots){
            state_manager.set("root" + to_string(i) + "_r", to_string(r.real()));
            state_manager.set("root" + to_string(i) + "_i", to_string(r.imag()));
            i++;
        }
        state_manager.set({{"roots_or_coefficients_control", "0"}});
    }

    vector<complex<float>> get_coefficients(){
        if(state["roots_or_coefficients_control"] != 0) {
            vector<complex<float>> coefficients;
            for(int point_index = 0; point_index < degree; point_index++) {
                string key = "coefficient" + to_string(point_index) + "_";
                coefficients.push_back(complex<float>(state[key + "r"], state[key + "i"]));
            }
            coefficients.push_back(1);
            return coefficients;
        }
        
        // The initial polynomial P(z) = 1
        vector<complex<float>> current_poly = {1.0};

        for (const auto& root : get_roots()) {
            // Each new polynomial will have one degree higher than the previous
            vector<complex<float>> new_poly(current_poly.size() + 1);

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
    vector<complex<float>> compute_derivative(const vector<complex<float>>& coefficients) {
        vector<complex<float>> derivative;
        int n = coefficients.size() - 1; // Degree of the polynomial

        for (int i = 1; i < n; ++i) {
            derivative.push_back(coefficients[i] * complex<float>(i, 0));
        }

        return derivative;
    }

    vector<complex<float>> deflate_polynomial(const vector<complex<float>>& coefficients, const complex<float>& root) {
        int n = coefficients.size() - 1;
        vector<complex<float>> new_coefficients(n);
        new_coefficients[0] = coefficients[0];
        for (int i = 1; i < n; ++i) {
            new_coefficients[i] = coefficients[i] + root * new_coefficients[i - 1];
        }
        return new_coefficients;
    }

    void stage_swap_roots_when_in_root_mode(TransitionType tt, const string& root1, const string& root2) {
        const string r0 = state_manager.get_equation("root" + root1 + "_r");
        const string i0 = state_manager.get_equation("root" + root1 + "_i");
        const string r1 = state_manager.get_equation("root" + root2 + "_r");
        const string i1 = state_manager.get_equation("root" + root2 + "_i");

        const string midpoint_r = r0 + " " + r1 + " + 2 /";
        const string midpoint_i = i0 + " " + i1 + " + 2 /";

        const string theta_unique = "theta"+to_string(rand());
        // Compute spin1 as (root1 - midpoint), rotate it about the origin by theta, then add midpoint back on

        // Define s1 = root1 - midpoint
        const string s1r = r0 + " " + midpoint_r + " -";
        const string s1i = i0 + " " + midpoint_i + " -";

        // Rotation formulas:
        // spin_r = cos(theta)*sXr - sin(theta)*sXi + midpoint_r
        // spin_i = sin(theta)*sXr + cos(theta)*sXi + midpoint_i

        const string cos_theta = "<" + theta_unique + "> cos";
        const string sin_theta = "<" + theta_unique + "> sin";

        const string spin_r1 = s1r + " " + cos_theta + " * " + s1i + " " + sin_theta + " * - " + midpoint_r + " +";
        const string spin_i1 = s1r + " " + sin_theta + " * " + s1i + " " + cos_theta + " * + " + midpoint_i + " +";

        const string spin_r2 = midpoint_r + " 2 * " + spin_r1 + " -";
        const string spin_i2 = midpoint_i + " 2 * " + spin_i1 + " -";

        state_manager.set(theta_unique, "0");
        state_manager.set({
            {"root" + root1 + "_r", spin_r1},
            {"root" + root1 + "_i", spin_i1},
            {"root" + root2 + "_r", spin_r2},
            {"root" + root2 + "_i", spin_i2},
        });
        state_manager.transition(tt, {
            {theta_unique, "pi"},
        });
        state_manager.transition(tt, {
            {"root" + root1 + "_r", r1},
            {"root" + root1 + "_i", i1},
            {"root" + root2 + "_r", r0},
            {"root" + root2 + "_i", i0},
        });
    }

    vector<complex<float>> get_roots(){
        vector<complex<float>> roots;
        if(state["roots_or_coefficients_control"] == 0) {
            for(int point_index = 0; point_index < degree; point_index++) {
                string key = "root" + to_string(point_index) + "_";
                roots.push_back(complex<float>(state[key + "r"], state[key + "i"]));
            }
        } else {
            vector<complex<float>> coefficients = get_coefficients();
            int n = coefficients.size() - 1;

            // Create the companion matrix
            MatrixXcd companion_matrix = MatrixXcd::Zero(n, n);
            for (int i = 0; i < n; ++i) {
                companion_matrix(i, n - 1) = -coefficients[i] / coefficients[n];
                if (i < n - 1) {
                    companion_matrix(i + 1, i) = complex<float>(1, 0);
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
                roots.push_back(complex<float>(eigenvalues[i].real(), eigenvalues[i].imag()));
            }
        }
        return roots;
    }

    void draw() override {
        const vector<complex<float>>& coefficients = get_coefficients();
        const vector<complex<float>>& roots = get_roots();
        int w = get_width();
        int h = get_height();

        // Decompose coefficients into separate real and imag arrays for CUDA call
        float h_coefficients_real[degree + 1];
        float h_coefficients_imag[degree + 1];
        for (int i = 0; i < coefficients.size(); ++i) {
            h_coefficients_real[i] = coefficients[i].real();
            h_coefficients_imag[i] = coefficients[i].imag();
        }

        color_complex_polynomial(
            pix.pixels.data(),
            pix.w,
            pix.h,
            h_coefficients_real,
            h_coefficients_imag,
            degree,
            state["left_x"], state["top_y"],
            state["right_x"], state["bottom_y"],
            state["ab_dilation"],
            state["dot_radius"]
        );

        float ro = state["roots_opacity"];
        if(ro > 0.01) {
            float gm = get_geom_mean_size() / 200;
            for(int i = 0; i < roots.size(); i++){
                float opa = ro * state["root"+to_string(i)+"_opacity"];
                if(opa < 0.01) continue;
                const glm::vec2 pixel(point_to_pixel(glm::vec2(roots[i].real(), roots[i].imag())));
                pix.fill_ring(pixel.x, pixel.y, gm*5, gm*4, OPAQUE_WHITE, opa);
            }
        }

        float co = state["coefficients_opacity"];
        if(co > 0.01) {
            for(int i = 0; i < coefficients.size()-1; i++){
                float opa = lerp(1, clamp(0,abs(coefficients[i])*2,1), state["hide_zero_coefficients"]);
                opa *= co * state["coefficient"+to_string(i)+"_opacity"];
                if(opa < 0.01) continue;
                const glm::vec2 pixel(point_to_pixel(glm::vec2(coefficients[i].real(), coefficients[i].imag())));
                ScalingParams sp = ScalingParams(get_width() / 10, get_height() / 10);
                Pixels text_pixels = latex_to_pix(string(1,char('a' + coefficients.size()-2-i)), sp);
                pix.overlay(text_pixels, pixel.x - text_pixels.w / 2, pixel.y - text_pixels.h / 2, opa);
            }
        }

        CoordinateScene::draw();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        for(string type : {"coefficient", "root"})
            for(int num = 0; num < degree; num++) {
                sq.insert(type + to_string(num) + "_opacity");
                for(char ri : {'r', 'i'})
                    sq.insert(type + to_string(num) + "_" + ri);
            }
        sq.insert("roots_or_coefficients_control");
        sq.insert("ab_dilation");
        sq.insert("dot_radius");
        sq.insert("roots_opacity");
        sq.insert("coefficients_opacity");
        sq.insert("hide_zero_coefficients");
        return sq;
    }
};
