#pragma once

#include "../Common/CoordinateScene.cpp"
#include "../../Host_Device_Shared/find_roots.c"

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

string new_coefficient_val = "0.001";

class ComplexPlotScene : public CoordinateScene {
private:
    vector<complex<float>> last_computed_roots;
    int degree;

    void decrement_degree() {
        cout << "Attempting to decrement degree from " << degree << endl;
        update_state();
        if(state["roots_or_coefficients_control"] != 1) {
            roots_to_coefficients();
        }

        if(degree <= 1) throw runtime_error("Cannot decrement degree below 1.");

        for(string ri : {"r", "i", "opacity"}) {
            string key = "coefficient" + to_string(degree) + "_" + ri;
            if(abs(state[key]) > stod(new_coefficient_val))
                throw runtime_error("Cannot decrement degree while leading coefficient is non-zero or non-opaque. Offending key: " + key + " with value " + to_string(state[key]));
            state.remove("root" + to_string(degree) + "_" + ri);
            state.remove("coefficient" + to_string(degree) + "_" + ri);
        }
        state.remove("coefficient" + to_string(degree) + "_ring");
        state.remove("root" + to_string(degree) + "_ring");
        state.remove("coefficient" + to_string(degree) + "_opacity");
        degree--;

        cout << "Decremented degree to " << degree << endl;
    }

    void increment_degree() {
        cout << "Attempting to increment degree from " << degree << endl;
        update_state();
        if(state["roots_or_coefficients_control"] != 1) {
            roots_to_coefficients();
        }
        cout << "Current leading coefficient: " << state["coefficient" + to_string(degree) + "_r"] << " + " << state["coefficient" + to_string(degree) + "_i"] << "i" << endl;

        degree++;

        for(string type : {"coefficient", "root"}) {
            string degree_str = to_string(degree - (type == "coefficient" ? 0 : 1));
            state.set(type + degree_str + "_r", new_coefficient_val);
            state.set(type + degree_str + "_i", "0");
            state.set(type + degree_str + "_ring", "0");
            if(type == "coefficient")
                state.set(type + degree_str + "_opacity", "0");
        }

        cout << "Incremented degree to " << degree << endl;
    }

public:
    ComplexPlotScene(const int d, const float width = 1, const float height = 1) : CoordinateScene(width, height), degree(d){
        complex_plane = true;
        for(string type : {"coefficient", "root"})
            for(int num = 0; num < (type == "coefficient"?degree+1:degree); num++){
                for(string ri : {"r", "i", "opacity", "ring"})
                    if(!(type == "root" && ri == "opacity"))
                        state.set(type + to_string(num) + "_" + ri, (ri == "opacity") ? "1" : "0");
            }
        state.set("roots_or_coefficients_control", "0"); // Default to root control
        state.set("ab_dilation", "1"); // basically saturation
        state.set("dot_radius", "1");
        state.set("positive_quadratic_formula_opacity", "0");
        state.set("negative_quadratic_formula_opacity", "0");
    }

    void transition_root_rings(const TransitionType tt, const float opacity) {
        for(int i = 0; i < degree; i++) {
            state.transition(tt, "root" + to_string(i) + "_ring", to_string(opacity));
        }
    }

    void transition_coefficient_rings(const TransitionType tt, const float opacity) {
        for(int i = 0; i < degree+1; i++) {
            state.transition(tt, "coefficient" + to_string(i) + "_ring", to_string(opacity));
        }
    }

    void transition_coefficient_opacities(const TransitionType tt, const float opacity) {
        for(int i = 0; i < degree+1; i++) {
            state.transition(tt, "coefficient" + to_string(i) + "_opacity", to_string(opacity));
        }
    }

    int get_degree() const {
        return degree;
    }

    void set_degree(int d) {
        if(d < 1) throw runtime_error("Degree must be at least 1. Requested degree: " + to_string(d));
        while(degree < d) increment_degree();
        while(degree > d) decrement_degree();
    }

    void roots_to_coefficients(){
        update_state();

        // If we are already in coefficients mode, do nothing
        if(state["roots_or_coefficients_control"] != 0) return;

        vector<complex<float>> coefficients = get_coefficients();

        int i = 0;
        for(complex<float> c : coefficients){
            state.set("coefficient" + to_string(i) + "_r", to_string(c.real()));
            state.set("coefficient" + to_string(i) + "_i", to_string(c.imag()));
            i++;
        }
        state.set({{"roots_or_coefficients_control", "1"}});
    }

    void coefficients_to_roots(){
        update_state();

        // If we are already in roots mode, do nothing
        if(state["roots_or_coefficients_control"] != 1) return;

        vector<complex<float>> roots = get_roots();
         
        cout << "Polynomial Coefficients:" << endl;
        for (complex<float> c : get_coefficients()) {
            cout << "Coefficient: " << c.real() << " + " << c.imag() << "i" << endl;
        }
        cout << "Computed Roots:" << endl;
        for (complex<float> r : roots) {
            cout << "Root: " << r.real() << " + " << r.imag() << "i" << endl;
        }

        int i = 0;
        for(complex<float> r : roots){
            state.set("root" + to_string(i) + "_r", to_string(r.real()));
            state.set("root" + to_string(i) + "_i", to_string(r.imag()));
            i++;
        }
        state.set({{"roots_or_coefficients_control", "0"}});
    }

    vector<complex<float>> get_coefficients() const {
        if(state["roots_or_coefficients_control"] != 0) {
            vector<complex<float>> coefficients;
            for(int point_index = 0; point_index < degree+1; point_index++) {
                string key = "coefficient" + to_string(point_index) + "_";
                coefficients.push_back(complex<float>(state[key + "r"], state[key + "i"]));
            }
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

    void stage_swap(TransitionType tt, const string& root1, const string& root2, bool root_or_coefficient, bool clockwise = false) {
        if(root_or_coefficient) roots_to_coefficients();
        else coefficients_to_roots();
        string type = root_or_coefficient ? "coefficient" : "root";
        const string r0 = state.get_equation(type + root1 + "_r");
        const string i0 = state.get_equation(type + root1 + "_i");
        const string r1 = state.get_equation(type + root2 + "_r");
        const string i1 = state.get_equation(type + root2 + "_i");

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

        state.set(theta_unique, "0");
        state.set({
            {type + root1 + "_r", spin_r1},
            {type + root1 + "_i", spin_i1},
            {type + root2 + "_r", spin_r2},
            {type + root2 + "_i", spin_i2},
        });
        state.transition(tt, theta_unique, clockwise ? "pi -1 *" : "pi");
        state.transition(tt, {
            {type + root1 + "_r", r1},
            {type + root1 + "_i", i1},
            {type + root2 + "_r", r0},
            {type + root2 + "_i", i0},
        });
    }

    void reorder_roots(vector<complex<float>>& roots) const {
        if(last_computed_roots.size() != roots.size()) return; // Can't reorder if sizes don't match

        vector<complex<float>> new_order(roots.size());

        for(int i = 0; i < last_computed_roots.size(); i++) {
            float min_dist = numeric_limits<float>::max();
            int min_index = -1;
            // Find the closest root to last_computed_roots[i]
            for(int j = 0; j < roots.size(); j++) {
                float dist = std::norm(roots[j] - last_computed_roots[i]);
                if(dist < min_dist) {
                    min_dist = dist;
                    min_index = j;
                }
            }
            if(min_index != -1) {
                new_order[i] = roots[min_index];
                // Remove this root from roots
                roots.erase(roots.begin() + min_index);
            }
        }
        roots = new_order;
    }

    vector<complex<float>> get_roots() const {
        vector<complex<float>> roots;
        if(state["roots_or_coefficients_control"] == 0) {
            for(int point_index = 0; point_index < degree; point_index++) {
                string key = "root" + to_string(point_index) + "_";
                roots.push_back(complex<float>(state[key + "r"], state[key + "i"]));
            }
        } else {
            vector<complex<float>> coefficients = get_coefficients();
            int n = coefficients.size() - 1;
            roots.resize(n);
            find_roots(get_coefficients().data(), n, roots.data());
            reorder_roots(roots); // Reorder roots to match last computed roots
        }
        return roots;
    }

    void draw() override {
        const vector<complex<float>>& coefficients = get_coefficients();
        const vector<complex<float>>& roots = get_roots();
        last_computed_roots = roots; // TODO This is awkward placement
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
            state["dot_radius"] * .4
        );

        float gm = get_geom_mean_size() / 200;

        // Draw roots
        for(int i = 0; i < roots.size(); i++){
            float opa = state["root"+to_string(i)+"_ring"];
            if(opa < 0.01) continue;
            const glm::vec2 pixel(point_to_pixel(glm::vec2(roots[i].real(), roots[i].imag())));
            pix.fill_ring(pixel.x, pixel.y, gm*6, gm*5, OPAQUE_WHITE, opa * .55);
        }

        // Draw coefficients
        for(int i = 0; i < coefficients.size(); i++){
            float ring_opa = state["coefficient"+to_string(i)+"_ring"];
            float letter_opa = state["coefficient"+to_string(i)+"_opacity"];
            const glm::vec2 pixel(point_to_pixel(glm::vec2(coefficients[i].real(), coefficients[i].imag())));
            if(ring_opa > 0.01) {
                pix.fill_ring(pixel.x, pixel.y, gm*6, gm*5, OPAQUE_WHITE, ring_opa * .55);
            }
            if(letter_opa > 0.01) {
                ScalingParams sp = ScalingParams(gm * 16, gm * 40);
                char letter = char('a' + coefficients.size() - 1 - i);
                Pixels text_pixels = latex_to_pix(string(1, letter), sp);

                // Align lowercase letters
                float align_factor = 0.5f;
                if(letter == 'b' || letter == 'd' || letter == 'f' || letter == 'h' || letter == 'k' || letter == 'l' || letter == 't') align_factor = 0.68f;
                else if(letter == 'g' || letter == 'j' || letter == 'p' || letter == 'q' || letter == 'y') align_factor = 0.32f;
                pix.overlay(text_pixels, pixel.x - text_pixels.w / 2, pixel.y - text_pixels.h * align_factor, letter_opa);
            }
        }

        // Draw quadratic formula roots
        for (int sign : {-1, 1}) {
            float opa = state[string(sign == 1 ? "posi" : "nega") + "tive_quadratic_formula_opacity"];
            if(opa > 0.01 && degree == 2) {
                complex<float> sign_c(sign, 0);
                const complex<float> a = coefficients[2];
                const complex<float> b = coefficients[1];
                const complex<float> c = coefficients[0];
                const complex<float> discriminant = b*b - complex<float>(4,0)*a*c;
                const complex<float> sqrt_disc = std::sqrt(discriminant);
                const complex<float> root = (-b + sqrt_disc * sign_c) / (2.0f*a);
                const glm::vec2 pixel(point_to_pixel(glm::vec2(root.real(), root.imag())));
                const int color = (sign == 1) ? 0xffff0000 : 0xff00ff00;
                pix.fill_ring(pixel.x, pixel.y, gm*6, gm*5, color, opa);
            }
        }

        CoordinateScene::draw();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        for(string type : {"coefficient", "root"})
            for(int num = 0; num < (type == "coefficient"?degree+1:degree); num++)
                for(string ri : {"r", "i", "opacity", "ring"})
                    if(!(type == "root" && ri == "opacity"))
                        sq.insert(type + to_string(num) + "_" + ri);
        state_query_insert_multiple(sq, {"roots_or_coefficients_control", "ab_dilation", "dot_radius", "positive_quadratic_formula_opacity", "negative_quadratic_formula_opacity"});
        return sq;
    }

    unordered_map<string, double> stage_publish_to_global() const override {
        unordered_map<string, double> map;

        vector<complex<float>> roots = get_roots();
        for(int i = 0; i < degree; i++) {
            map["root" + to_string(i) + "_r"] = roots[i].real();
            map["root" + to_string(i) + "_i"] = roots[i].imag();
            map["root" + to_string(i) + "_ring"] = state["root" + to_string(i) + "_ring"];
        }

        vector<complex<float>> coefficients = get_coefficients();
        for(int i = 0; i < degree+1; i++) {
            map["coefficient" + to_string(i) + "_r"] = coefficients[i].real();
            map["coefficient" + to_string(i) + "_i"] = coefficients[i].imag();
            map["coefficient" + to_string(i) + "_opacity"] = state["coefficient" + to_string(i) + "_opacity"];
            map["coefficient" + to_string(i) + "_ring"] = state["coefficient" + to_string(i) + "_ring"];
        }

        return map;
    }
};
