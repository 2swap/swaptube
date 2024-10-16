#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <complex>

extern "C" void mandelbrot_render(
    const int width, const int height,
    const complex<double> seed_z, const complex<double> seed_x, const complex<double> seed_c,
    const glm::vec3 pixel_parameter_multipliers,
    const complex<double> offset, const complex<double> zoom,
    int max_iterations,
    unsigned int* depths
);

class MandelbrotScene : public Scene {
public:
    MandelbrotScene(const double width = 1, const double height = 1) : Scene(width, height) { }
    const StateQuery populate_state_query() const override {
        return StateQuery{"zoom_r", "zoom_i", "offset_r", "offset_i", "max_iterations", "seed_z_r", "seed_z_i", "seed_x_r", "seed_x_i", "seed_c_r", "seed_c_i", "pixel_param_z", "pixel_param_x", "pixel_param_c"};
    }

    void on_end_transition() override {}
    void mark_data_unchanged() override {}
    void change_data() override {}
    bool check_if_data_changed() const override {return false;}
    void draw() override {
        glm::vec3 pixel_params = glm::vec3(state["pixel_param_z"], state["pixel_param_x"], state["pixel_param_c"]);
        mandelbrot_render(pix.w, pix.h,
                          complex(state["seed_z_r"], state["seed_z_i"]),
                          complex(state["seed_x_r"], state["seed_x_i"]),
                          complex(state["seed_c_r"], state["seed_c_i"]),
                          glm::normalize(pixel_params),
                          complex(state["offset_r"], state["offset_i"]), complex(state["zoom_r"], state["zoom_i"]),
                          state["max_iterations"],
                          pix.pixels.data()
        );
    }

};

