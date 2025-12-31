#pragma once

#include "../Common/CoordinateScene.cpp"
#include <vector>
#include <stdexcept>
#include <string>

inline string replace_substring(string str, const string& from, const string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Move past the replacement to avoid infinite loop
    }
    return str;
}

class RealFunctionScene : public CoordinateScene {
public:
    RealFunctionScene(const double width = 1, const double height = 1) : CoordinateScene(width, height) {
        manager.set({
            {"function0", "0"},
            {"function1", "0"},
            {"function0_opacity", "1"},
            {"function1_opacity", "0"},
            {"ticks_opacity", "1"},
        });
    }

    // Evaluates the function at x by replacing '{a}' with the x-value,
    // computing the current function and its transitional variant, then smoothing between them.
    float call_the_function(const float x, const ResolvedStateEquation& re) {
        int error;
        float y = evaluate_resolved_state_equation(re.size(), re.data(), &x, 1, error);
        if(error != 0)
            throw runtime_error("Error evaluating function in RealFunctionScene::call_the_function");
        return y;
    }

    void draw() override {
        render_functions();
        CoordinateScene::draw();
    }

    void render_functions() {
        double left_bound  = state["left_x"];
        double right_bound = state["right_x"];
        double dx = (right_bound - left_bound) / static_cast<double>(get_width());
        double last_subtr = 0;

        vector<ResolvedStateEquation> resolved_equations;
        vector<float> opacities;
        for(int i = 0; i < 2; i++) {
            resolved_equations.push_back(manager.get_resolved_equation("function" + to_string(i)));
            opacities.push_back(state["function" + to_string(i) + "_opacity"]);
        }

        float thickness = get_geom_mean_size() / 200.f;
        for (int func_idx = 0; func_idx < 2; func_idx++) {
            float opacity = opacities[func_idx];
            if (opacity <= 0.01) continue;

            const ResolvedStateEquation& re = resolved_equations[func_idx];
            glm::vec2 last_pixel = {-1, -1};
            bool first_point = true;
            int color = function_colors[func_idx];
            for (double x = left_bound; x <= right_bound; x += dx) {
                float val = call_the_function(x, re);
                glm::vec2 pixel = point_to_pixel({x, val});
                if (first_point) {
                    first_point = false;
                    last_pixel = pixel;
                    continue;
                }
                pix.bresenham(pixel.x, pixel.y, last_pixel.x, last_pixel.y, color, opacity, thickness);
                last_pixel = pixel;
            }
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        state_query_insert_multiple(sq, {"function0", "function1", "function0_opacity", "function1_opacity"});
        return sq;
    }

    void mark_data_unchanged() override {}
    void change_data() override {} // RealFunctionScene has no DataObjects
    bool check_if_data_changed() const override { return false; }

private:
    vector<unsigned int> function_colors = {0xFF0088FF, 0xFFFF0088};
};

