#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Scene.cpp"
#include <math.h>

class PendulumGridScene : public Scene {
public:
    PendulumGridScene(const double init_p1, const double init_p2, const double width = 1, const double height = 1) : Scene(width, height), grid(get_width(), get_height(), init_p1, init_p2) {}

    const StateQuery populate_state_query() const override {
        return StateQuery{"physics_multiplier", "rk4_step_size"};
    }

    void on_end_transition() override {}

    void mark_data_unchanged() override { grid.mark_unchanged(); }
    void change_data() override { grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]); }
    bool check_if_data_changed() const override { return grid.has_been_updated_since_last_scene_query(); }

    void draw() override {
        int w = get_width();
        int h = get_height();

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double theta1 = grid.pendulum_states[y][x].theta1;
                double theta2 = grid.pendulum_states[y][x].theta2;
                double thetap1 = grid.pendulum_pairs[y][x].theta1;
                double thetap2 = grid.pendulum_pairs[y][x].theta2;

                //int color = YUVtoRGB(map_to_torus(theta1, theta2));
                double distance = square(theta1-thetap1) + square(theta2-thetap2);
                int color = distance > 0.1 ? OPAQUE_WHITE : OPAQUE_BLACK;
                pix.set_pixel(x, y, color);
            }
        }
    }

private:
    PendulumGrid grid;
};

