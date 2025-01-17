#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"

void butterfly(const double sx, const double sy) {
    CompositeScene cs;
    vector<PendulumScene> vps;
    for(int i = 0; i < 9; i++){
        PendulumState pendulum_state = {sx+(i/3-1)*.01, sy + (i%3-1) * .01, .0, .0};
        PendulumScene ps(pendulum_state);
        ps.state_manager.set({
            {"background_opacity", "0"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "16"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        vps.push_back(ps);
    }
    for(int i = 0; i < vps.size(); i++){
        string key = "ps" + to_string(i);
        cs.add_scene(&(vps[i]), key);
    }
    cs.inject_audio_and_render(SilenceSegment(10));
}

void grid() {
    CompositeScene cs;
    vector<PendulumScene> vps;
    int gridsize = 41;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            PendulumState pendulum_state = {(x-gridsize/2)*.1, (y-gridsize/2)*.1, .0, .0};
            PendulumScene ps(pendulum_state, gridstep*1.02, gridstep*1.02);
            StateSet state = {
                {"pendulum_opacity",   "[pendulum_opacity]"  },
                {"background_opacity", "[background_opacity]"},
                {"physics_multiplier", "[physics_multiplier]"},
                {"rk4_step_size",      "[rk4_step_size]"     },
            };
            ps.state_manager.set(state);
            vps.push_back(ps);
        }
    }
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            string key = "ps" + to_string(x+y*gridsize);
            cs.add_scene(&(vps[x+y*gridsize]), key, gridstep*(x+.5), gridstep*(y+.5));
        }
    }
    StateSet state = {
        {"pendulum_opacity", "0"},
        {"background_opacity", "1"},
        {"physics_multiplier", "1"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    };
    cs.state_manager.set(state);
    cs.inject_audio_and_render(SilenceSegment(2));
}

void fractal() {
    PendulumGridScene pgs(0, 0);
    pgs.state_manager.set({
        {"physics_multiplier", "32"},
        {"mode", "1"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 40 /"},
    });
    pgs.inject_audio_and_render(SilenceSegment(5));
}

void lissajous() {
    vector<PendulumState> ps;
    ps.push_back({0.3, .3, .0, .0});
    ps.push_back({3.6, 3, .0, .0});
    ps.push_back({2.49, .25, .0, .0});
    ps.push_back({2, .3, .0, .0});
    ps.push_back({0.1, .3, .0, .0});
    vector<double> zooms{0.5, .01, .02, .01, 0.5};
    for(int i = 0; i < ps.size(); i++){
        vector<float> left;
        vector<float> right;
        CompositeScene cs;
        PendulumScene this_ps(ps[i], 0.5, 0.5);
        this_ps.global_publisher_key = true;
        this_ps.state_manager.set({
            {"physics_multiplier", "16"},
            {"pendulum_opacity", "1"},
            {"background_opacity", "0.1"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        CoordinateScene coord;
        coord.state_manager.set({
            {"center_x", "0"},
            {"center_y", "0"},
            {"zoom", to_string(zooms[i])},
            {"trail_x", "{pendulum_theta1}"},
            {"trail_y", "{pendulum_theta2}"},
        });
        cs.add_scene(&this_ps, "this_ps", 0.25, 0.25);
        cs.add_scene(&coord, "coord", 0.5, 0.5);
        this_ps.generate_audio(6, left, right);
        cs.inject_audio_and_render(GeneratedSegment(left, right));
    }
}

void render_video() {
    PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;

    //butterfly(1, 1.25);
    //butterfly(2.49, .25);
    lissajous();
    //grid();
    //fractal();
}
