#pragma once

#include "../Common/CoordinateScene.cpp"
#include "ComplexPlotScene.cpp" // Contains definition of populate_roots
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

extern "C" void draw_root_fractal(unsigned int* pixels, int w, int h, complex<float> c1, complex<float> c2, float terms, float lx, float ty, float rx, float by, float radius, float opacity);

class RootFractalScene : public CoordinateScene {
public:
    RootFractalScene(const float width = 1, const float height = 1) : CoordinateScene(width, height) {
        state_manager.set("coefficient0_r", "0");
        state_manager.set("coefficient0_i", "0");
        state_manager.set("coefficient1_r", "1");
        state_manager.set("coefficient1_i", "0");
        state_manager.set("terms", "8");
        state_manager.set("degree_fixed", "1");
        state_manager.set("coefficients_opacity", "1");
    }

    void draw() override {
        int w = get_width();
        int h = get_height();
        complex<float> c0(state["coefficient0_r"], state["coefficient0_i"]);
        complex<float> c1(state["coefficient1_r"], state["coefficient1_i"]);
        // Take the ceiling of the degree to ensure it's an integer
        float n = state["terms"];
        float radius = get_geom_mean_size() * pow(state["zoom"]*5, .25) / 150;
        float opacity = 1-1/(2*state["zoom"]+1);
        draw_root_fractal(pix.pixels.data(), w, h,
            c0, c1, n,
            state["left_x"], state["top_y"],
            state["right_x"], state["bottom_y"],
            radius, opacity
        );

        float gm = get_geom_mean_size() / 200;

        // Draw coefficients
        int i = 0;
        for(complex<float> coeff : {c0, c1}) {
            float letter_opa = 1;
            letter_opa *= state["coefficients_opacity"];
            const glm::vec2 pixel(point_to_pixel(glm::vec2(coeff.real(), coeff.imag())));
            if(letter_opa > 0.01) {
                ScalingParams sp = ScalingParams(gm * 16, gm * 40);
                Pixels text_pixels = latex_to_pix(string(1,char('a' + i)), sp);
                pix.overlay(text_pixels, pixel.x - text_pixels.w / 2, pixel.y - text_pixels.h / 2, letter_opa);
            }
            i++;
        }
        CoordinateScene::draw();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        sq.insert("coefficient0_r");
        sq.insert("coefficient0_i");
        sq.insert("coefficient1_r");
        sq.insert("coefficient1_i");
        sq.insert("terms");
        sq.insert("zoom");
        sq.insert("degree_fixed");
        sq.insert("left_x");
        sq.insert("top_y");
        sq.insert("right_x");
        sq.insert("bottom_y");
        sq.insert("coefficients_opacity");
        return sq;
    }
};
