#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Common/CoordinateScene.cpp"
#include "../Scene.cpp"
#include <math.h>

class MovingPendulumGridScene : public CoordinateScene {
public:
    MovingPendulumGridScene(const double width = 1, const double height = 1) : CoordinateScene(width, height) {
        state_manager.add_equation("contrast", ".1");
        state_manager.add_equation("mode", "0");
        state_manager.add_equation("center_x", "0");
        state_manager.add_equation("center_y", "0");
        state_manager.add_equation("theta_or_momentum", "0");
        state_manager.add_equation("physics_multiplier", "0");
        state_manager.add_equation("theta1", "0");
        state_manager.add_equation("theta2", "0");
        state_manager.add_equation("p1", "0");
        state_manager.add_equation("p2", "0");
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = CoordinateScene::populate_state_query();
        s.insert("physics_multiplier");
        s.insert("rk4_step_size");
        s.insert("mode");
        s.insert("center_x");
        s.insert("center_y");
        s.insert("contrast");
        s.insert("theta_or_momentum");
        s.insert("theta1");
        s.insert("theta2");
        s.insert("p1");
        s.insert("p2");
        return s;
    }

    void mark_data_unchanged() override { }
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }

    void draw_grid() {
        int w = get_width();
        int h = get_height();

        const double cx = state["center_x"];
        const double cy = state["center_y"];
        const double contrast = state["contrast"];
        const double mode = state["mode"];

        const double coloration = 1000;
        const double log_coloration = log(coloration);
        const double tom = state["theta_or_momentum"];

        PendulumGrid grid(w, h,
            state["left_x"  ] * (1-tom) + state["theta1"],
            state["right_x" ] * (1-tom) + state["theta1"],
            state["bottom_y"] * (1-tom) + state["theta2"],
            state["top_y"   ] * (1-tom) + state["theta2"],
            state["left_x"  ] * (  tom) + state["p1"    ],
            state["right_x" ] * (  tom) + state["p1"    ],
            state["bottom_y"] * (  tom) + state["p2"    ],
            state["top_y"   ] * (  tom) + state["p2"    ]
        );
        grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int i = x+y*w;

                int color_mode0 = 0; int color_mode1 = 0; int color_mode2 = 0; int color_mode3 = 0;
                int color = 0xffff0000;

                if(mode < 1.999) color_mode0 = pendulum_color(grid.pendulum_states[i].theta1, grid.pendulum_states[i].theta2);
                if(mode > 0.001 && mode < 1.999) color_mode1 = colorlerp(color_mode0, OPAQUE_WHITE, max(0.,log(grid.diff_sums[i]*contrast)/5));
                if(mode > 1.001 && mode < 2.999) color_mode2 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0.,log(grid.diff_sums[i]*contrast)/5));
                if(mode > 2.001) {
                    PendulumState ps = grid.pendulum_states[i];
                    PendulumState pp = grid.pendulum_pairs[i];
                    
                    double distance = sqrt(square(ps.p1 - pp.p1) + square(ps.p2 - pp.p2) + square(ps.theta1-pp.theta1) + square(ps.theta2-pp.theta2));
                    distance = min(distance, 1.);
                    color_mode3 = colorlerp(OPAQUE_BLACK, OPAQUE_WHITE, max(0., log(coloration*distance)/log_coloration));
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
        CoordinateScene::draw();
    }
};

