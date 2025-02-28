#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Common/CoordinateScene.cpp"
#include "../Scene.cpp"
#include <math.h>

class PendulumGridScene : public CoordinateScene {
public:
    PendulumGridScene(const vector<PendulumGrid>& pgv, const double width = 1, const double height = 1) : CoordinateScene(width, height), grids(pgv) {
        state_manager.add_equation("contrast", "1");
        state_manager.add_equation("mode", "0");
        state_manager.add_equation("physics_multiplier", "0");
        state_manager.add_equation("rk4_step_size", "1 30 / .1 *");
        state_manager.add_equation("zoom", "1 6.283 /");
        state_manager.add_equation("center_x", "0");
        state_manager.add_equation("center_y", "0");
        state_manager.add_equation("trail_start_x", "0");
        state_manager.add_equation("trail_start_y", "0");
        state_manager.add_equation("trail_opacity", "0");
        state_manager.add_equation("trail_length", "0");
        state_manager.add_equation("energy_min", "0");
        state_manager.add_equation("energy_max", "0");
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
        s.insert("energy_min");
        s.insert("energy_max");
        return s;
    }

    void mark_data_unchanged() override {
        for(PendulumGrid& grid : grids)
            grid.mark_unchanged();
    }
    void change_data() override {
        for(PendulumGrid& grid : grids)
            grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]);
    }
    bool check_if_data_changed() const override {
        for(const PendulumGrid& grid : grids)
            if(grid.has_been_updated_since_last_scene_query())
                return true;
        return false;
    }

    void draw_grid() {
        int w = get_width();
        int h = get_height();

        const double zoomy = state["zoom_y"];
        const double zoomx = state["zoom_x"];
        const double cx = state["center_x"];
        const double cy = state["center_y"];
        const double contrast = state["contrast"];
        const double mode = state["mode"];
        const double emin = state["energy_min"];
        const double emax = state["energy_max"];

        const double coloration = 1000;
        const double log_coloration = log(coloration);

        const PendulumGrid& g0 = grids[0];
        const double inv_y_range = 1./(g0.max2-g0.min2);
        const double inv_x_range = 1./(g0.max1-g0.min1);
        for (int y = 0; y < h; ++y) {
            double pos_y = (h/2.0 - y) / (h * zoomy) + cy;
            pos_y = extended_mod(pos_y-g0.min2, g0.max2-g0.min2)+g0.min2;
            for (int x = 0; x < w; ++x) {
                double pos_x = (x - w/2.0) / (w * zoomx) + cx;
                pos_x = extended_mod(pos_x-g0.min1, g0.max1-g0.min1)+g0.min1;

                int last_grid = 0;
                for(int i = grids.size() - 1; i >= 0; i--){
                    const PendulumGrid& grid = grids[i];
                    if(pos_x < grid.max1 && pos_x > grid.min1 && pos_y < grid.max2 && pos_y > grid.min2){
                        last_grid = i;
                        break;
                    }
                }
                const PendulumGrid& grid = grids[last_grid];
                int arr_x = grid.w * (pos_x - grid.min1) / (grid.max1-grid.min1);
                if(arr_x < 0 || arr_x >= grid.w) continue;
                int arr_y = grid.h * (pos_y - grid.min2) / (grid.max2-grid.min2);
                if(arr_y < 0 || arr_y >= grid.h) continue;
                int i = arr_x+arr_y*grid.w;

                int color_mode0 = 0; int color_mode1 = 0; int color_mode2 = 0; int color_mode3 = 0;
                int color = 0xffff0000;

                double how_chaotic = max(0.,grid.diff_sums[i]/grid.samples*contrast);
                if(mode < 1.999) color_mode0 = pendulum_color(grid.pendulum_states[i].theta1, grid.pendulum_states[i].theta2);
                if(mode > 0.001 && mode < 1.999) color_mode1 = colorlerp(color_mode0, OPAQUE_WHITE, how_chaotic);
                if(mode > 1.001 && mode < 2.999) color_mode2 = black_to_blue_to_white(how_chaotic);
                if(mode > 2.001) {
                    PendulumState ps = grid.pendulum_states[i];
                    PendulumState pp = grid.pendulum_pairs[i];
                    
                    double distance = sqrt(square(ps.p1 - pp.p1) + square(ps.p2 - pp.p2) + square(ps.theta1-pp.theta1) + square(ps.theta2-pp.theta2));
                    distance = min(distance, 1.);
                    double rainbow = max(0., log(coloration*distance)/log_coloration);
                    color_mode3 = black_to_blue_to_white(rainbow);
                }

                if(mode < 0.001) color = color_mode0;
                else if(mode > 0.999 && mode < 1.001) color = color_mode1;
                else if(mode <= 0.999 && mode >= 0.001) color = colorlerp(color_mode0, color_mode1, mode);
                else if(mode > 1.999 && mode < 2.001) color = color_mode2;
                else if(mode <= 1.999 && mode >= 1.001) color = colorlerp(color_mode1, color_mode2, mode-1);
                else if(mode > 2.999) color = color_mode3;
                else if(mode <= 2.999 && mode >= 2.001) color = colorlerp(color_mode2, color_mode3, mode-2);
                if(emax>0.01) {
                    double energy = compute_potential_energy(grid.start_states[i]) + compute_kinetic_energy(grid.start_states[i]);
                    if(energy>emin && energy<emax) {
                        color = colorlerp(color, 0xffff0000, 0.5);
                    }
                }
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

    bool momentum_mode = false;
private:
    vector<PendulumGrid> grids;
};

