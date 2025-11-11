#pragma once

#include "../Common/CoordinateScene.cpp"
#include <glm/glm.hpp>
#include <complex>

extern "C" void mandelbrot_render(
    const int width, const int height,
    const complex<float> seed_z, const complex<float> seed_x, const complex<float> seed_c,
    const glm::vec3 pixel_parameter_multipliers,
    const complex<float> zoom,
    int max_iterations,
    float gradation,
    float phase_shift,
    unsigned int internal_color,
    unsigned int* depths
);

class MandelbrotScene : public CoordinateScene {
public:
    MandelbrotScene(const double width = 1, const double height = 1) : CoordinateScene(width, height) {
        manager.set({
            {"max_iterations", "32"},
            {"seed_z_r", "0"},
            {"seed_z_i", "0"},
            {"seed_x_r", "2"},
            {"seed_x_i", "0"},
            {"seed_c_r", "0"},
            {"seed_c_i", "0"},
            {"pixel_param_z", "0"}, // Julia set
            {"pixel_param_x", "0"}, // X set
            {"pixel_param_c", "1"}, // Mandelbrot set
            {"point_path_length", "0"},
            {"point_path_start_r", "0"},
            {"point_path_start_i", "0"},
            {"gradation", "1"},
            {"phase_shift", "{t}"},
        });
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        state_query_insert_multiple(sq, {"max_iterations", "seed_z_r", "seed_z_i", "seed_x_r", "seed_x_i", "seed_c_r", "seed_c_i", "pixel_param_z", "pixel_param_x", "pixel_param_c", "point_path_length", "point_path_start_r", "point_path_start_i", "gradation", "phase_shift", "zoom"});
        return sq;
    }

    void mark_data_unchanged() override {}
    void change_data() override {}
    bool check_if_data_changed() const override {return false;}
    void draw() override {
        glm::vec3 pixel_params = glm::normalize(glm::vec3(state["pixel_param_z"], state["pixel_param_x"], state["pixel_param_c"]));
        complex<float> seed_z(state["seed_z_r"], state["seed_z_i"]);
        complex<float> seed_x(state["seed_x_r"], state["seed_x_i"]);
        complex<float> seed_c(state["seed_c_r"], state["seed_c_i"]);
        mandelbrot_render(pix.w, pix.h,
                          seed_z, seed_x, seed_c,
                          pixel_params,
                          state["zoom"], // TODO use native functions and integrate better with CoordinateScene, or is it futile?
                          state["max_iterations"],
                          state["gradation"],
                          state["phase_shift"],
                          OPAQUE_BLACK,
                          pix.pixels.data()
        );
        if(state["point_path_length"] > 0) {
            // TODO convert to use CoordinateSceneWithTrail
            int startcol = 0xffff0000;
            int pathcol = 0xff880000;
            complex<float> z(state["point_path_start_r"], state["point_path_start_i"]);
            glm::vec2 start = point_to_pixel(glm::vec2(state["point_path_start_r"], state["point_path_start_i"]));
            float r = get_width()/300.;
            float thickness = get_width()/960.;
            pix.fill_circle(start.x, start.y, r, startcol);
            int iter_count = state["point_path_length"];
            for(int i = 0; i < iter_count; i++){
                complex<float> new_z = pow(z, seed_x) + seed_c;
                glm::vec2 prev = point_to_pixel(glm::vec2(z.real(), z.imag()));
                glm::vec2 next = point_to_pixel(glm::vec2(new_z.real(), new_z.imag()));
                pix.fill_circle(next.x, next.y, r, pathcol);
                pix.bresenham(prev.x, prev.y, next.x, next.y, pathcol, 1, thickness);
                z = new_z;
                if(abs(z) > 100) return;
            }
        }
        CoordinateScene::draw();
    }
};
