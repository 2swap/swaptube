#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <complex>

extern "C" void mandelbrot_render(
    const int width, const int height,
    const complex<double> seed_z, const complex<double> seed_x, const complex<double> seed_c,
    const glm::vec3 pixel_parameter_multipliers,
    const complex<double> zoom,
    int max_iterations,
    unsigned int* depths
);

class MandelbrotScene : public Scene {
public:
    MandelbrotScene(const double width = 1, const double height = 1) : Scene(width, height) { }
    const StateQuery populate_state_query() const override {
        return StateQuery{"zoom_r", "zoom_i", "max_iterations", "seed_z_r", "seed_z_i", "seed_x_r", "seed_x_i", "seed_c_r", "seed_c_i", "pixel_param_z", "pixel_param_x", "pixel_param_c", "side_panel"};
    }

    void on_end_transition() override {}
    void mark_data_unchanged() override {}
    void change_data() override {}
    bool check_if_data_changed() const override {return false;}
    void draw() override {
        glm::vec3 pixel_params = glm::normalize(glm::vec3(state["pixel_param_z"], state["pixel_param_x"], state["pixel_param_c"]));
        complex seed_z = complex(state["seed_z_r"], state["seed_z_i"]);
        complex seed_x = complex(state["seed_x_r"], state["seed_x_i"]);
        complex seed_c = complex(state["seed_c_r"], state["seed_c_i"]);
        complex zoom = complex(state["zoom_r"], state["zoom_i"]);
        int main_panel_w = lerp(pix.w, pix.w*3/4, state["side_panel"]);
        Pixels main_panel(main_panel_w, pix.h);
        mandelbrot_render(main_panel.w, main_panel.h,
                          seed_z, seed_x, seed_c,
                          pixel_params,
                          zoom,
                          state["max_iterations"],
                          main_panel.pixels.data()
        );
        pix.overwrite(main_panel, 0, 0);
        if(state["side_panel"] > 0.01) {
            int side_panel_widths = pix.w - main_panel.w;
            int remaining_height = pix.h;
            int remaining_panels = 3;
            for(int i = 0; i < 3; i++){
                Pixels panel(side_panel_widths, remaining_height/remaining_panels);
                mandelbrot_render(panel.w, panel.h,
                                  seed_z, seed_x, seed_c,
                                  glm::vec3(i==0,i==1,i==2),
                                  1,
                                  state["max_iterations"],
                                  panel.pixels.data()
                );
                pix.overwrite(panel, main_panel.w, pix.h - remaining_height);
                remaining_height -= panel.h;
                remaining_panels--;
            }
        }
    }

};

