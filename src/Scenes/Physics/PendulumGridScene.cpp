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
        state_manager.add_equation("trail_start_x", "0");
        state_manager.add_equation("trail_start_y", "0");
        state_manager.add_equation("trail_opacity", "0");
        state_manager.add_equation("trail_length", "0");
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = CoordinateScene::populate_state_query();
        s.insert("physics_multiplier");
        s.insert("rk4_step_size");
        s.insert("mode");
        s.insert("center_x");
        s.insert("center_y");
        s.insert("contrast");
        s.insert("trail_opacity");
        s.insert("trail_length");
        s.insert("trail_start_y");
        s.insert("trail_start_x");
        return s;
    }

    void mark_data_unchanged() override { grid.mark_unchanged(); }
    void change_data() override { grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]); }
    bool check_if_data_changed() const override { return grid.has_been_updated_since_last_scene_query(); }

    void draw_grid() {
        int w = get_width();
        int h = get_height();

        const double zoom = state["zoom"];
        const double cx = state["center_x"];
        const double cy = state["center_y"];
        const double contrast = state["contrast"];
        const double mode = state["mode"];

        const double inv_y_range = 1./(max_y-min_y);
        const double inv_x_range = 1./(max_x-min_x);
        for (int y = 0; y < h; ++y) {
            double pos_y = (h/2.0 - y) / (h * zoom) + cy;
            int arr_y = static_cast<int>(((pos_y-min_y)*inv_y_range+100) * grid.h)%grid.h;
            if(arr_y >= grid.h || arr_y < 0) continue;
            for (int x = 0; x < w; ++x) {
                double pos_x = (x - w/2.0) / (w * zoom) + cx;
                int arr_x = static_cast<int>(((pos_x-min_x)*inv_x_range+100) * grid.w)%grid.w;
                if(arr_x >= grid.w || arr_x < 0) continue;

                int i = arr_x + arr_y * grid.w;

                int color_mode0 = 0; int color_mode1 = 0; int color_mode2 = 0; int color_mode3 = 0;
                int color = 0xffff0000;

                if(mode < 1.999) color_mode0 = colorlerp(OPAQUE_BLACK, YUVtoRGB(map_to_torus(grid.pendulum_states[i].theta1, grid.pendulum_states[i].theta2)), 0.5);
                if(mode > 0.001 && mode < 1.999) color_mode1 = colorlerp(color_mode0, OPAQUE_WHITE, max(0.,log(grid.diff_sums[i]*contrast)/5));
                if(mode > 1.001 && mode < 2.999) color_mode2 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0.,log(grid.diff_sums[i]*contrast)/5));
                if(mode > 2.001) {
                    PendulumState ps = grid.pendulum_states[i];
                    PendulumState pp = grid.pendulum_pairs[i];
                    
                    double distance = sqrt(square(ps.p1 - pp.p1) + square(ps.p2 - pp.p2) + square(ps.theta1-pp.theta1) + square(ps.theta2-pp.theta2));
                    distance = min(distance, 1.);
                    color_mode3 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0., log(distance)));
                }

                if(mode < 0.001) color = color_mode0;
                else if(mode > 0.999 && mode < 1.001) color = color_mode1;
                else if(mode <= 0.999 && mode >= 0.001) color = colorlerp(color_mode0, color_mode1, mode);
                else if(mode > 1.999 && mode < 2.001) color = color_mode2;
                else if(mode <= 1.999 && mode >= 1.001) color = colorlerp(color_mode1, color_mode2, mode-1);
                else if(mode > 2.999) color = color_mode3;
                else if(mode <= 2.999 && mode >= 2.001) color = colorlerp(color_mode2, color_mode3, mode-2);
                pix.set_pixel(x, y, color);
            }
        }
    }

    void draw() override {
        draw_grid();
        draw_pendulum_trail();
        CoordinateScene::draw();
    }

    void draw_pendulum_trail(){
        const double trail_opacity = state["trail_opacity"];
        if(trail_opacity < 0.01) return;
        const double trail_start_x = state["trail_start_x"];
        const double trail_start_y = state["trail_start_y"];
        const double trail_length = state["trail_length"];
        Pendulum p({trail_start_x, trail_start_y, 0, 0});
        vector<pair<double, double>> trail;
        for(int i = 0; i < trail_length; i++){
            p.iterate_physics(1, 0.01);
            trail.push_back(make_pair(p.state.theta1, p.state.theta2));
        }
        draw_trail(trail, 0xffff0000, trail_opacity);
        draw_point(make_pair(trail_start_x, trail_start_y), 0xffff0000, trail_opacity);
    }

private:
    PendulumGrid grid;
    const double min_x;
    const double max_x;
    const double min_y;
    const double max_y;
};

