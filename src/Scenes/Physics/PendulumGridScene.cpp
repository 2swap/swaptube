#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Common/CoordinateScene.cpp"
#include "../Scene.cpp"
#include <math.h>

class PendulumGridScene : public CoordinateScene {
public:
    PendulumGridScene(const double init_p1, const double init_p2, const double width = 1, const double height = 1) : CoordinateScene(width, height), grid(get_width(), get_height(), init_p1, init_p2), sums(get_width() * get_height()) {}

    const StateQuery populate_state_query() const override {
        StateQuery s = CoordinateScene::populate_state_query();
        s.insert("physics_multiplier");
        s.insert("rk4_step_size");
        s.insert("mode");
        s.insert("center_x");
        s.insert("center_y");
        return s;
    }

    void on_end_transition() override {}

    void mark_data_unchanged() override { grid.mark_unchanged(); }
    void change_data() override { cout << "Physicsing..."; grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]); cout << "Done!" << endl; }
    bool check_if_data_changed() const override { return grid.has_been_updated_since_last_scene_query(); }

    void draw() override {
        cout << "Drawing...";
        int w = get_width();
        int h = get_height();

        const double zoom = state["zoom"];
        const double cx = state["center_x"];
        const double cy = state["center_y"];

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double pos_x = (x - w/2.0) / (w * zoom) + cx;
                double pos_y = (h/2.0 - y) / (h * zoom) + cy;
                int arr_x = (w*50 + static_cast<int>(pos_x * w / (20./*M_PI*2*/))) % w;
                int arr_y = (h*50 + static_cast<int>(pos_y * h / (20.))) % h;
                int i = arr_x + arr_y * w;

                double theta1  = grid.pendulum_states[i].theta1;
                double theta2  = grid.pendulum_states[i].theta2;
                double thetap1 = grid.pendulum_pairs [i].theta1;
                double thetap2 = grid.pendulum_pairs [i].theta2;

                int color_mode0 = colorlerp(OPAQUE_BLACK, YUVtoRGB(map_to_torus(theta1, theta2)), 0.5);
                double distance = square(theta1-thetap1) + square(theta2-thetap2);
                distance = min(distance, .01);
                sums[i] += distance;
                int color_mode1 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0.,log(sums[i]*10000)/15));
                //int color_mode1 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0., log(distance*100)/10));
                int color = colorlerp(color_mode0, color_mode1, state["mode"]);

                pix.set_pixel(x, y, color);
            }
        }
        CoordinateScene::draw();
        cout << "Done!" << endl;
    }

private:
    PendulumGrid grid;
    vector<double> sums;
};

