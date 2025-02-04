#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Common/CoordinateScene.cpp"
#include "../Scene.cpp"
#include <math.h>

class PendulumGridScene : public CoordinateScene {
public:
    PendulumGridScene(const double min_x, const double max_x, const double min_y, const double max_y, const PendulumGrid& pg, const double width = 1, const double height = 1) : CoordinateScene(width, height), grid(pg), min_x(min_x), max_x(max_x), min_y(min_y), max_y(max_y) {
        state_manager.add_equation("contrast", ".1");
        state_manager.add_equation("mode", "0");
        state_manager.add_equation("center_x", "0");
        state_manager.add_equation("center_y", "0");
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = CoordinateScene::populate_state_query();
        s.insert("physics_multiplier");
        s.insert("rk4_step_size");
        s.insert("mode");
        s.insert("center_x");
        s.insert("center_y");
        s.insert("contrast");
        return s;
    }

    void on_end_transition() override {}

    void mark_data_unchanged() override { grid.mark_unchanged(); }
    void change_data() override { cout << "Physicsing..." << endl; grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]); cout << "Done!" << endl; }
    bool check_if_data_changed() const override { return grid.has_been_updated_since_last_scene_query(); }

    void draw() override {
        cout << "Drawing..." << endl;
        int w = get_width();
        int h = get_height();

        const double zoom = state["zoom"];
        const double cx = state["center_x"];
        const double cy = state["center_y"];
        const double contrast = state["contrast"];

        for (int y = 0; y < h; ++y) {
            double pos_y = (h/2.0 - y) / (h * zoom) + cy;
            int arr_y = static_cast<int>(((pos_y-min_y)/max_y+100) * grid.h)%grid.h;
            if(arr_y >= grid.h || arr_y < 0) continue;
            for (int x = 0; x < w; ++x) {
                double pos_x = (x - w/2.0) / (w * zoom) + cx;
                int arr_x = static_cast<int>(((pos_x-min_x)/max_x+100) * grid.w)%grid.w;
                if(arr_x >= grid.w || arr_x < 0) continue;

                int i = arr_x + arr_y * grid.w;

                int color_mode0 = 0; int color_mode1 = 0;
                double mode = state["mode"];
                if(mode < 0.999) color_mode0 = colorlerp(OPAQUE_BLACK, YUVtoRGB(map_to_torus(grid.pendulum_states[i].theta1, grid.pendulum_states[i].theta2)), 0.5);
                if(mode > 0.001) color_mode1 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0.,log(grid.diff_sums[i]*contrast)/6));
                int color = colorlerp(color_mode0, color_mode1, mode);
                pix.set_pixel(x, y, color);
            }
        }
        CoordinateScene::draw();
        cout << "Done!" << endl;
    }

private:
    PendulumGrid grid;
    const double min_x;
    const double max_x;
    const double min_y;
    const double max_y;
};

