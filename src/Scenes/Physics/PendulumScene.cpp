#include "PendulumScene.h"
#include <cmath>
#include <algorithm>

PendulumScene::PendulumScene(PendulumState s, const vec2& dimensions) : Scene(dimensions), start_state(s) {
    path_background = Pixels(floor(get_width_height()));
    manager.set({{"tone", "1"},
                       {"volume", "0"},
                       {"path_opacity", "0"},
                       {"physics_multiplier", "30"},
                       {"rk4_step_size", "1 30 / <physics_multiplier> 0.01 + /"},
                       {"rainbow", "1"}});
    pend = new Pendulum(s);
    add_data_object(pend);
}

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);
extern "C" void draw_quadrilateral(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const vec2& p3, const uint32_t color);

const StateQuery PendulumScene::populate_state_query() const {
    return StateQuery{"volume", "rainbow", "tone", "path_opacity", "physics_multiplier", "rk4_step_size"};
}

void PendulumScene::draw() {
    double w = get_width(); double h = get_height();
    double line_thickness = h/40;
    vec2 pos = get_width_height()*.5;
    std::vector<double> thetas = {pend->state.theta1, pend->state.theta2};
    int color = pendulum_color(thetas[0], thetas[1], pend->state.p1, pend->state.p2);
    //if(state["path_opacity"] > 0.01)
    //    pix.overlay_gpu(path_background, 0, 0);

    float rainbow = state["rainbow"];
    uint32_t pendulum_color = colorlerp(OPAQUE_WHITE, color, rainbow);
    double length = h/(pendulum_count * 2 + 1.);
    for (int i = 0; i < pendulum_count; i++) {
        double theta = thetas[i];
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        vec2 delta = length * vec2(sin_theta, cos_theta);
        draw_circle(gpu_pix->get_ptr(), get_width_height(), pos, line_thickness*1.3, pendulum_color);
        vec2 lateral = vec2(cos_theta, -sin_theta)*line_thickness/2;
        vec2 p0 = pos + lateral;
        vec2 p1 = pos - lateral;
        pos += delta;
        vec2 p2 = pos - lateral;
        vec2 p3 = pos + lateral;
        draw_quadrilateral(gpu_pix->get_ptr(), get_width_height(), p0, p1, p2, p3, pendulum_color);
    }
    draw_circle(gpu_pix->get_ptr(), get_width_height(), pos, line_thickness*1.3, pendulum_color);
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
