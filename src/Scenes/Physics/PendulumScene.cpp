#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Scene.cpp"

class PendulumScene : public Scene {
public:
    PendulumScene(PendulumState s, int c, const double width = 1, const double height = 1) : Scene(width, height), start_state(s), pend(s), color(c) { }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

    void on_end_transition() override {}
    void mark_data_unchanged() override { pend.mark_unchanged(); }
    void change_data() override { pend.iterate_physics_once(); }
    bool check_if_data_changed() const override { return pend.has_been_updated_since_last_scene_query(); }

    void draw() override {
        double w = get_width(); double h = get_height();
        double line_thickness = h/60;
        double posx = w/2; double posy = h/2;
        vector<double> thetas = {pend.state.theta1, pend.state.theta2};
        int pendulum_count = 2;
        for (int i = 0; i < pendulum_count; i++) {
            double theta = thetas[i];
            int divider = pendulum_count * 2 + 1;
            double dx = sin(theta) * h / divider; double dy = cos(theta) * h / divider;
            pix.fill_circle(posx, posy, line_thickness * 2, color);
            pix.bresenham(posx, posy, posx + dx, posy + dy, color, 1, line_thickness);
            posx += dx; posy += dy;
        }
        pix.fill_circle(posx, posy, line_thickness*2, color);
    }

    void generate_audio(double duration, vector<float>& left, vector<float>& right){
        PendulumState state = start_state;
        for(int i = 0; i < duration*44100; i++){
            state = rk4Step(state, 0.03);
            left.push_back(sin(state.theta1));
            right.push_back(sin(state.theta2));
        }
    }

private:
    PendulumState start_state;
    Pendulum pend;
    int color;
};
