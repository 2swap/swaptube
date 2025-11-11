#pragma once

#include "../Common/CoordinateScene.cpp"

extern "C" void draw_root_fractal(unsigned int* pixels, int w, int h, complex<float> c1, complex<float> c2, float terms, float lx, float ty, float rx, float by, float radius, float opacity, float brightness);

class RootFractalScene : public CoordinateScene {
public:
    RootFractalScene(const float width = 1, const float height = 1) : CoordinateScene(width, height) {
        state.set("coefficient0_r", "-1");
        state.set("coefficient0_i", "0");
        state.set("coefficient1_r", "1");
        state.set("coefficient1_i", "0");
        state.set("terms", "8");
        state.set("degree_fixed", "1");
        state.set("coefficients_opacity", "1");
        state.set("visibility_multiplier", "1");
        state.set("brightness", ".25");
    }

    void draw() override {
        int w = get_width();
        int h = get_height();
        complex<float> c0(state["coefficient0_r"], state["coefficient0_i"]);
        complex<float> c1(state["coefficient1_r"], state["coefficient1_i"]);
        // Take the ceiling of the degree to ensure it's an integer
        float n = state["terms"];
        float wh = state["window_height"];
        float radius = sqrt(state["visibility_multiplier"]) * get_geom_mean_size() * pow(wh*10, .25) / 250;
        float opacity = 1-1/(2*wh+1);
        opacity *= square(state["visibility_multiplier"]);
        draw_root_fractal(pix.pixels.data(), w, h,
            c0, c1, n,
            state["left_x"], state["top_y"],
            state["right_x"], state["bottom_y"],
            radius, opacity, state["brightness"]
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

    unordered_map<string, double> stage_publish_to_global() const override {
        return unordered_map<string, double> {
            {"coefficient0_r", state["coefficient0_r"]},
            {"coefficient0_i", state["coefficient0_i"]},
            {"coefficient1_r", state["coefficient1_r"]},
            {"coefficient1_i", state["coefficient1_i"]},
        };
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        state_query_insert_multiple(sq, {"coefficient0_r", "coefficient0_i", "coefficient1_r", "coefficient1_i", "terms", "window_height", "degree_fixed", "left_x", "top_y", "right_x", "bottom_y", "coefficients_opacity", "visibility_multiplier", "brightness"});
        return sq;
    }
};
