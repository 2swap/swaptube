#include "RealFunctionScene.h"
#include <vector>
#include <stdexcept>
#include <string>

RealFunctionScene::RealFunctionScene(const double width, const double height) : CoordinateScene(width, height) {
    manager.set({
        {"function0", "0"},
        {"function1", "0"},
        {"function0_left", "-100000"},
        {"function0_right", "100000"},
        {"function1_left", "-100000"},
        {"function1_right", "100000"},
        {"function0_opacity", "1"},
        {"function1_opacity", "0"},
        {"ticks_opacity", "1"},
    });
}

// Evaluates the function at x by replacing '{a}' with the x-value,
// computing the current function and its transitional variant, then smoothing between them.
float RealFunctionScene::call_the_function(const float x, const ResolvedStateEquation& re) {
    int error;
    float y = evaluate_resolved_state_equation(re.size(), re.data(), &x, 1, error);
    if(error != 0)
        throw runtime_error("Error evaluating function in RealFunctionScene::call_the_function");
    return y;
}

void RealFunctionScene::draw() {
    render_functions();
    CoordinateScene::draw();
}

void RealFunctionScene::render_functions() {
    double left_bound  = state["left_x"];
    double right_bound = state["right_x"];
    double dx = (right_bound - left_bound) / static_cast<double>(get_width());
    double last_subtr = 0;

    std::vector<ResolvedStateEquation> resolved_equations;
    std::vector<float> opacities;
    for(int i = 0; i < 2; i++) {
        resolved_equations.push_back(manager.get_resolved_equation("function" + to_string(i)));
        opacities.push_back(state["function" + to_string(i) + "_opacity"]);
    }

    float thickness = get_geom_mean_size() / 200.f;
    for (int func_idx = 0; func_idx < 2; func_idx++) {
        float left_bound_this = max(left_bound, state["function" + to_string(func_idx) + "_left"]);
        float right_bound_this = min(right_bound, state["function" + to_string(func_idx) + "_right"]);
        if (left_bound_this >= right_bound_this) continue;
        float opacity = opacities[func_idx];
        if (opacity <= 0.01) continue;

        const ResolvedStateEquation& re = resolved_equations[func_idx];
        glm::vec2 last_pixel = {-1, -1};
        bool first_point = true;
        int color = function_colors[func_idx];
        for (double x = left_bound_this; x <= right_bound_this; x += dx) {
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

const StateQuery RealFunctionScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {"function0", "function1", "function0_opacity", "function1_opacity", "function0_left", "function0_right", "function1_left", "function1_right"});
    return sq;
}
