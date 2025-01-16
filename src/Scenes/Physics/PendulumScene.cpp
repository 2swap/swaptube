#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Scene.cpp"

class PendulumScene : public Scene {
public:
    PendulumScene(PendulumState s, const double width = 1, const double height = 1) : Scene(width, height), start_state(s), pend(s) { }

    const StateQuery populate_state_query() const override {
        return StateQuery{"physics_multiplier", "rk4_step_size", "pendulum_opacity", "background_opacity"};
    }

    void on_end_transition() override {}
    void mark_data_unchanged() override { pend.mark_unchanged(); }
    void change_data() override { pend.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]); }
    bool check_if_data_changed() const override { return pend.has_been_updated_since_last_scene_query(); }
    unordered_map<string, double> stage_publish_to_global() const override {
        return unordered_map<string, double> {
            {"pendulum_theta1", pend.state.theta1},
            {"pendulum_theta2", pend.state.theta2},
        };
    }

    void draw() override {
        double w = get_width(); double h = get_height();
        double line_thickness = h/60;
        double posx = w/2; double posy = h/2;
        vector<double> thetas = {pend.state.theta1, pend.state.theta2};
        int pendulum_count = 2;
        int color = YUVtoRGB(map_to_torus(thetas[0], thetas[1]));
        //pix.fill(colorlerp(OPAQUE_BLACK, color, state["background_opacity"]));

        if(state["pendulum_opacity"] > 0.01) {
            int pendulum_color = colorlerp(OPAQUE_BLACK, color, state["pendulum_opacity"]);
            for (int i = 0; i < pendulum_count; i++) {
                double theta = thetas[i];
                int divider = pendulum_count * 2 + 1;
                double dx = sin(theta) * h / divider; double dy = cos(theta) * h / divider;
                pix.fill_circle(posx, posy, line_thickness * 2, pendulum_color);
                pix.bresenham(posx, posy, posx + dx, posy + dy, pendulum_color, 1, line_thickness);
                posx += dx; posy += dy;
            }
            pix.fill_circle(posx, posy, line_thickness*2, pendulum_color);
        }
    }

    void generate_audio(double duration, vector<float>& left, vector<float>& right){
        PendulumState state = start_state;
        for(int i = 0; i < duration*44100; i++){
            for(int j = 0; j < 10; j++) {
                state = rk4Step(state, 0.003);
            }
            left.push_back(sin(state.theta1));
            right.push_back(sin(state.theta2));
        }
    }

private:
    PendulumState start_state;
    Pendulum pend;
};
