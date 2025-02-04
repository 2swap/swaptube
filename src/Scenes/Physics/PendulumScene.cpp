#pragma once
#include "../../DataObjects/Pendulum.cpp"
#include "../Scene.cpp"

class PendulumScene : public Scene {
public:
    PendulumScene(PendulumState s, const double width = 1, const double height = 1) : Scene(width, height), start_state(s), pend(s), path_background(get_width(), get_height()) {
        state_manager.add_equation("tone", "1");
        state_manager.add_equation("volume", "0");
        state_manager.add_equation("path_opacity", "0");
        state_manager.add_equation("angles_opacity", "0");
        state_manager.add_equation("rainbow", "1");
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"angles_opacity", "volume", "rainbow", "tone", "path_opacity", "t", "physics_multiplier", "rk4_step_size", "pendulum_opacity", "background_opacity"};
    }

    void on_end_transition() override {}
    void mark_data_unchanged() override { pend.mark_unchanged(); }
    void change_data() override {
        for(int i = 0; i < state["physics_multiplier"]; i++) {
            pend.iterate_physics(1, state["rk4_step_size"]);
            energy_slew = square(compute_kinetic_energy(pend.state));
            generate_tone();
        }
    }
    bool check_if_data_changed() const override { return pend.has_been_updated_since_last_scene_query(); }
    unordered_map<string, double> stage_publish_to_global() const override {
        return unordered_map<string, double> {
            {"pendulum_theta1", pend.state.theta1},
            {"pendulum_theta2", pend.state.theta2},
        };
    }

    void draw() override {
        double w = get_width(); double h = get_height();
        double line_thickness = h/80;
        double posx = w/2; double posy = h/2;
        vector<double> thetas = {pend.state.theta1, pend.state.theta2};
        int pendulum_count = 2;
        int color = YUVtoRGB(map_to_torus(thetas[1], thetas[0]));
        if(state["background_opacity"] > 0.01)
            pix.fill(colorlerp(TRANSPARENT_BLACK, color, state["background_opacity"]));

        if(state["pendulum_opacity"] > 0.01) {
            double rainbow = state["rainbow"];
            int pendulum_color = OPAQUE_WHITE;
            if(rainbow > 0.01)
                pendulum_color = colorlerp(OPAQUE_WHITE, colorlerp(TRANSPARENT_BLACK, color, state["pendulum_opacity"]), rainbow);
            for (int i = 0; i < pendulum_count; i++) {
                double theta = thetas[i];
                double length = h/(pendulum_count * 2 + 1.);
                double dx = sin(theta) * length; double dy = cos(theta) * length;
                pix.fill_circle(posx, posy, line_thickness * 2, pendulum_color);
                pix.bresenham(posx, posy, posx + dx, posy + dy, pendulum_color, 1, line_thickness);
                double ao = state["angles_opacity"];
                if(ao > 0.01){
                    double theta_modified = theta+199*M_PI;
                    theta_modified -= static_cast<int>(theta_modified/(2*M_PI))*2*M_PI + M_PI;
                    pix.bresenham(posx, posy, posx, posy + length, OPAQUE_WHITE, ao, .5*line_thickness);
                    for(double angle = 0; angle < 1; angle+=.01) {
                        pix.overlay_pixel(posx + sin(angle*theta_modified)*length*.5, posy + cos(angle*theta_modified)*length*.5, OPAQUE_WHITE, ao);
                    }
                }
                posx += dx; posy += dy;
            }
            pix.fill_circle(posx, posy, line_thickness*2, pendulum_color);
        }
        if(state["path_opacity"] > 0.01 && (last_posx != 0 || last_posy != 0)) {
            path_background.bresenham(last_posx, last_posy, posx, posy, OPAQUE_WHITE, state["path_opacity"], line_thickness/4.);
            pix.underlay(path_background, 0, 0);
        }
        last_posx = posx; last_posy = posy;
        for(int x = 0; x < path_background.w; x++) {
            for(int y = 0; y < path_background.h; y++) {
                int alpha = geta(path_background.get_pixel(x, y));
                alpha = alpha==0?0:alpha-1;
                path_background.set_pixel(x, y, argb_to_col(alpha, 255, 255, 255));
            }
        }
    }

    void generate_tone(){
        double vol = state["volume"];
        if(vol < 0.01) return;
        if(tonegen == 0) tonegen = state["t"]*44100;
        vector<float> left;
        vector<float> right;
        int total_samples = 44100/VIDEO_FRAMERATE/state["physics_multiplier"];
        int tonegen_save = tonegen;
        double note = state["tone"];
        for(int i = 0; i < total_samples; i++){
            double strength = lerp(energy, energy_slew, static_cast<double>(i)/total_samples);
            float val = .00002*vol*strength*sin(tonegen*2200.*note/44100.)/sqrt(note);
            tonegen++;
            left.push_back(val);
            right.push_back(val);
        }
        WRITER.add_sfx(left, right, tonegen_save);
        energy = energy_slew;
    }

    void generate_audio(double duration, vector<float>& left, vector<float>& right){
        PendulumState ps = start_state;
        for(int i = 0; i < duration*44100; i++){
            for(int j = 0; j < 10; j++) {
                ps = rk4Step(ps, 0.003);
            }
            left.push_back(.1*sin(ps.theta1));
            right.push_back(.1*sin(ps.theta2));
        }
    }

private:
    int tonegen = 0;
    double energy = 0;
    double energy_slew = 0;
    double last_posx; double last_posy;
    PendulumState start_state;
    Pendulum pend;
    Pixels path_background;
};
