#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    CompositeScene cs;
    vector<PendulumScene> vps;
    int gridsize = 41;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            PendulumState pendulum_state = {(x-gridsize/2)*.1, (y-gridsize/2)*.1, .0, .0};
            PendulumScene ps(pendulum_state, 0xffffffff, gridstep*1.02, gridstep*1.02);
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
    cs.inject_audio_and_render(SilenceSegment(7));
}
