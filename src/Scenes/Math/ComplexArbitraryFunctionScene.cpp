#pragma once

#include "../Common/CoordinateScene.cpp"

// This is a sorta one-time use scene for the Quintic video for rendering complex plots of sqrt, sin, exp, etc.

extern "C" void color_complex_arbitrary_function(
    unsigned int* h_pixels, // to be overwritten with the result
    int w,
    int h,
    float sqrt_coef, float sqrt_branch_cut, float sin_coef, float cos_coef, float exp_coef,
    float lx, float ty, float rx, float by,
    float ab_dilation,
    float dot_radius
);

class ComplexArbitraryFunctionScene : public CoordinateScene {
public:
    ComplexArbitraryFunctionScene(const float width = 1, const float height = 1) : CoordinateScene(width, height) {
        state.set("sqrt_coef", "1");
        state.set("sqrt_branch_cut", "1");
        state.set("sin_coef", "0");
        state.set("cos_coef", "0");
        state.set("exp_coef", "0");
        state.set("ab_dilation", "1");
        state.set("dot_radius", "1");
    }

    void draw() override {
        int w = get_width();
        int h = get_height();

        // Draw the function
        color_complex_arbitrary_function(
            pix.pixels.data(), w, h,
            state["sqrt_coef"],
            state["sqrt_branch_cut"],
            state["sin_coef"],
            state["cos_coef"],
            state["exp_coef"],
            state["left_x"], state["top_y"],
            state["right_x"], state["bottom_y"],
            state["ab_dilation"],
            state["dot_radius"] * .4
        );

        CoordinateScene::draw();
    }

    unordered_map<string, double> stage_publish_to_global() const override {
        return unordered_map<string, double> {
            {"sqrt_branch_cut", state["sqrt_branch_cut"]},
        };
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        state_query_insert_multiple(sq, {"sqrt_coef", "sqrt_branch_cut", "sin_coef", "cos_coef", "exp_coef", "ab_dilation", "dot_radius"});
        return sq;
    }
};
