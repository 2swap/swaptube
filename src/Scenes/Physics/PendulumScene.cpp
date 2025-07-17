#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Scene.cpp"

class PendulumScene : public Scene {
public:
    PendulumScene(PendulumState s, const double width = 1, const double height = 1) : Scene(width, height), pend(s), start_state(s), path_background(get_width(), get_height()) {
        state_manager.set({{"tone", "1"},
                           {"volume", "0"},
                           {"path_opacity", "0"},
                           {"physics_multiplier", "30"},
                           {"rk4_step_size", "1 30 / <physics_multiplier> 0.01 + /"},
                           {"background_opacity", "0"},
                           {"pendulum_opacity", "1"},
                           {"top_angle_opacity", "0"},
                           {"bottom_angle_opacity", "0"},
                           {"rainbow", "1"},
                           {"manual_mode", "0"},
                           {"theta1_manual", "0"},
                           {"theta2_manual", "0"}});
    }
    int alpha_subtract = 2;

    const StateQuery populate_state_query() const override {
        return StateQuery{"manual_mode", "theta1_manual", "theta2_manual", "top_angle_opacity", "bottom_angle_opacity", "volume", "rainbow", "tone", "path_opacity", "t", "physics_multiplier", "rk4_step_size", "pendulum_opacity", "background_opacity"};
    }

    void mark_data_unchanged() override { pend.mark_unchanged(); }
    void change_data() override {
        double w = get_width(); double h = get_height();
        double line_thickness = h/80;
        double length = h/(pendulum_count * 2 + 1.);
        double pm = state["physics_multiplier"];
        double last_x = w/2 + (sin(pend.state.theta1)+sin(pend.state.theta2))*length;
        double last_y = h/2 + (cos(pend.state.theta1)+cos(pend.state.theta2))*length;
        for(int i = 0; i < pm; i++) {
            pend.iterate_physics(1, state["rk4_step_size"]);
            double x = w/2 + (sin(pend.state.theta1)+sin(pend.state.theta2))*length;
            double y = h/2 + (cos(pend.state.theta1)+cos(pend.state.theta2))*length;
            path_background.bresenham(last_x, last_y, x, y, OPAQUE_WHITE, state["path_opacity"], line_thickness/4.);
            last_x = x; last_y = y;
        }
        energy_slew = square(compute_kinetic_energy(pend.state))/100.;
        generate_tone();
        for(int x = 0; x < path_background.w; x++) {
            for(int y = 0; y < path_background.h; y++) {
                int alpha = geta(path_background.get_pixel(x, y));
                alpha = max(0,alpha-alpha_subtract);
                path_background.set_pixel(x, y, argb(alpha, 255, 255, 255));
            }
        }
    }
    bool check_if_data_changed() const override { return pend.has_been_updated_since_last_scene_query(); }
    unordered_map<string, double> stage_publish_to_global() const override {
        return unordered_map<string, double> {
            {"pendulum_p1", pend.state.p1},
            {"pendulum_p2", pend.state.p2},
            {"pendulum_theta1", pend.state.theta1},
            {"pendulum_theta2", pend.state.theta2},
        };
    }

    void draw() override {
        double w = get_width(); double h = get_height();
        double line_thickness = h/80;
        double posx = w/2; double posy = h/2;
        double in_manual_mode = state["manual_mode"];
        vector<double> thetas = {lerp(pend.state.theta1, state["theta1_manual"], in_manual_mode),
                                 lerp(pend.state.theta2, state["theta2_manual"], in_manual_mode)};
        int color = pendulum_color(thetas[0], thetas[1], pend.state.p1, pend.state.p2);
        if(state["background_opacity"] > 0.01)
            pix.fill(colorlerp(TRANSPARENT_BLACK, color, state["background_opacity"]));

        double pend_opa = state["pendulum_opacity"];
        if(pend_opa > 0.01) {
            double rainbow = state["rainbow"];
            int pendulum_color = colorlerp(OPAQUE_WHITE, color, rainbow);
            for (int i = 0; i < pendulum_count; i++) {
                double theta = thetas[i];
                double length = h/(pendulum_count * 2 + 1.);
                double dx = sin(theta) * length; double dy = cos(theta) * length;
                pix.fill_circle(posx, posy, line_thickness * 2, pendulum_color, pend_opa);
                pix.bresenham(posx, posy, posx + dx, posy + dy, pendulum_color, pend_opa, line_thickness);
                double ao = i==0?state["top_angle_opacity"]:state["bottom_angle_opacity"];
                if(ao > 0.01){
                    double theta_modified = theta+199*M_PI;
                    theta_modified -= static_cast<int>(theta_modified/(2*M_PI))*2*M_PI + M_PI;
                    pix.bresenham(posx, posy, posx, posy + length, OPAQUE_WHITE, ao, .5*line_thickness);
                    const double d_angle = .01;
                    for(double angle = 0; angle < 1; angle+=d_angle) {
                        pix.bresenham(posx + sin( angle         *theta_modified)*length*.5,
                                      posy + cos( angle         *theta_modified)*length*.5,
                                      posx + sin((angle-d_angle)*theta_modified)*length*.5,
                                      posy + cos((angle-d_angle)*theta_modified)*length*.5, OPAQUE_WHITE, ao, .25*line_thickness);
                    }
                }
                posx += dx; posy += dy;
            }
            pix.fill_circle(posx, posy, line_thickness*2, pendulum_color, pend_opa);
        }
        if(state["path_opacity"] > 0.01) {
            pix.underlay(path_background, 0, 0);
        }
    }

    void generate_tone(){
        double vol = state["volume"];
        int total_samples = SAMPLERATE/FRAMERATE;
        if(vol < 0.01) {tonegen = 0; return;}
        if(tonegen == 0) tonegen = state["t"]*SAMPLERATE;
        vector<float> left;
        vector<float> right;
        int tonegen_save = tonegen;
        double note = state["tone"];
        for(int i = 0; i < total_samples; i++){
            double strength = lerp(energy, energy_slew, static_cast<double>(i)/total_samples);
            float val = .002*vol*strength*sin(tonegen*2200.*note/SAMPLERATE)/sqrt(note);
            tonegen++;
            left.push_back(val);
            right.push_back(val);
        }
        AUDIO_WRITER.add_sfx(left, right, tonegen_save);
        energy = energy_slew;
    }

    void generate_audio(double duration, vector<float>& left, vector<float>& right, double volume_mult = 1){
        PendulumState ps = start_state;
        for(int i = 0; i < duration*SAMPLERATE; i++){
            for(int j = 0; j < 10; j++) {
                ps = rk4Step(rk4Step(rk4Step(ps, 0.001), 0.001), 0.001);
            }
            left.push_back(.05*volume_mult*sin(ps.theta1));
            right.push_back(.05*volume_mult*sin(ps.theta2));
        }
    }
    Pendulum pend;

private:
    int tonegen = 0;
    double energy = 0;
    double energy_slew = 0;
    PendulumState start_state;
    Pixels path_background;
    const int pendulum_count = 2;
};
