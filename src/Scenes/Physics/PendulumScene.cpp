#include "PendulumScene.h"
#include <cmath>
#include <algorithm>

PendulumScene::PendulumScene(PendulumState s, const vec2& dimensions) : Scene(dimensions), pend(s), start_state(s) {
    path_background = Pixels(get_width(), get_height());
    manager.set({{"tone", "1"},
                       {"volume", "0"},
                       {"path_opacity", "0"},
                       {"physics_multiplier", "30"},
                       {"rk4_step_size", "1 30 / <physics_multiplier> 0.01 + /"},
                       {"pendulum_opacity", "1"},
                       {"top_angle_opacity", "0"},
                       {"bottom_angle_opacity", "0"},
                       {"rainbow", "1"},
                       {"manual_mode", "0"},
                       {"theta1_manual", "0"},
                       {"theta2_manual", "0"}});
    add_data_object(&pend);
}

const StateQuery PendulumScene::populate_state_query() const {
    return StateQuery{"manual_mode", "theta1_manual", "theta2_manual", "top_angle_opacity", "bottom_angle_opacity", "volume", "rainbow", "tone", "path_opacity", "physics_multiplier", "rk4_step_size", "pendulum_opacity"};
}

std::unordered_map<std::string, double> PendulumScene::stage_publish_to_global() const {
    return std::unordered_map<std::string, double> {
        {"pendulum_p1", pend.state.p1},
        {"pendulum_p2", pend.state.p2},
        {"pendulum_theta1", pend.state.theta1},
        {"pendulum_theta2", pend.state.theta2},
    };
}

void PendulumScene::draw() {
    double w = get_width(); double h = get_height();
    double line_thickness = h/80;
    double posx = w/2; double posy = h/2;
    double in_manual_mode = state["manual_mode"];
    std::vector<double> thetas = {lerp(pend.state.theta1, state["theta1_manual"], in_manual_mode),
                                 lerp(pend.state.theta2, state["theta2_manual"], in_manual_mode)};
    int color = pendulum_color(thetas[0], thetas[1], pend.state.p1, pend.state.p2);
    if(state["path_opacity"] > 0.01)
        pix.overlay(path_background, 0, 0);

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
}

void PendulumScene::generate_tone(){
    double vol = state["volume"];
    int total_samples = get_samples_per_frame();
    if(vol < 0.01) {tonegen = 0; return;}
    if(tonegen == 0) tonegen = state["t"]*get_audio_samplerate_hz();
    std::vector<sample_t> left;
    std::vector<sample_t> right;
    int tonegen_save = tonegen;
    double note = state["tone"];
    for(int i = 0; i < total_samples; i++){
        double strength = lerp(energy, energy_slew, static_cast<double>(i)/total_samples);
        float val_f = .002*vol*strength*sin(tonegen*2200.*note/get_audio_samplerate_hz())/sqrt(note);
        sample_t val = float_to_sample(val_f);
        tonegen++;
        left.push_back(val);
        right.push_back(val);
    }
    get_writer().audio->add_sfx(left, right, tonegen_save);
    energy = energy_slew;
}

void PendulumScene::generate_audio(double duration, std::vector<sample_t>& left, std::vector<sample_t>& right, double volume_mult){
    PendulumState ps = start_state;
    for(int i = 0; i < duration*get_audio_samplerate_hz(); i++) {
        for(int j = 0; j < 10; j++) {
            ps = rk4Step(rk4Step(rk4Step(ps, 0.001), 0.001), 0.001);
        }
        left.push_back(float_to_sample(.05*volume_mult*sin(ps.theta1)));
        right.push_back(float_to_sample(.05*volume_mult*sin(ps.theta2)));
    }
}
