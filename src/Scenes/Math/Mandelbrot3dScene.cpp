#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <complex>

extern "C" void mandelbrot_3d_render(
    const int width, const int height,
    const complex<double> seed_z, const complex<double> seed_x, const complex<double> seed_c,
    int max_iterations,
    unsigned int internal_color,
    unsigned int* depths
);

class Mandelbrot3dScene : public Scene {
public:
    Mandelbrot3dScene(const double width = 1, const double height = 1) : Scene(width, height) {
        state_manager.set({{"max_iterations", "50"},
                           {"seed_z_r", "0"},
                           {"seed_z_i", "0"},
                           {"seed_x_r", "2"},
                           {"seed_x_i", "0"},
                           {"seed_c_r", "0"},
                           {"seed_c_i", "0"}});
    }
    const StateQuery populate_state_query() const override {
        return StateQuery{"max_iterations", "seed_z_r", "seed_z_i", "seed_x_r", "seed_x_i", "seed_c_r", "seed_c_i"};
    }

    void mark_data_unchanged() override {}
    void change_data() override {}
    bool check_if_data_changed() const override {return false;}

    void draw() override {
        unsigned int internal_color = 0xff0000ff;
        complex seed_z(state["seed_z_r"], state["seed_z_i"]);
        complex seed_x(state["seed_x_r"], state["seed_x_i"]);
        complex seed_c(state["seed_c_r"], state["seed_c_i"]);
        mandelbrot_3d_render(pix.w, pix.h,
                          seed_z, seed_x, seed_c,
                          state["max_iterations"],
                          internal_color,
                          pix.pixels.data()
        );
    }
};
