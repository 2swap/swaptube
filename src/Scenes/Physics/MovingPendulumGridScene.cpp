#include "../../DataObjects/Pendulum.h"
#include "../Common/CoordinateScene.h"
#include <math.h>

class MovingPendulumGridScene : public CoordinateScene {
public:
    MovingPendulumGridScene(const double width = 1, const double height = 1) : CoordinateScene(width, height) {
        manager.set({{"contrast", ".1"},
                {"mode", "0"},
                {"center_x", "0"},
                {"center_y", "0"},
                {"theta_or_momentum", "0"},
                {"physics_multiplier", "0"},
                {"theta1", "0"},
                {"theta2", "0"},
                {"p1", "0"},
                {"p2", "0"},
                {"momentum_value_gradient", "1"}});
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = CoordinateScene::populate_state_query();
        state_query_insert_multiple(s, {"physics_multiplier", "rk4_step_size", "mode", "center_x", "center_y", "contrast", "theta_or_momentum", "theta1", "theta2", "p1", "p2", "momentum_value_gradient"});
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
        const double mvg = state["momentum_value_gradient"];

        const double coloration = 1000;
        const double log_coloration = log(coloration);
        const double tom = state["theta_or_momentum"];

        PendulumGrid grid(w, h, 0.0001,
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

                double how_chaotic = max(0.,grid.diff_sums[i]/grid.samples*contrast);
                double use_p1 = grid.pendulum_states[i].p1 * mvg;
                double use_p2 = grid.pendulum_states[i].p2 * mvg;
                if(mode < 1.999) color_mode0 = pendulum_color(grid.pendulum_states[i].theta1, grid.pendulum_states[i].theta2, use_p1, use_p2);
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
                pix.set_pixel_carelessly(x, y, color);
            }
        }
    }

    void draw() override {
        draw_grid();
        CoordinateScene::draw();
    }
};

