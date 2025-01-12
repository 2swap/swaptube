#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    CompositeScene cs;
    vector<PendulumScene> vps;
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {2.49+(i/3-1)*.01, .25 + (i%3-1) * .01, .0, .0};
        PendulumScene ps(pendulum_state);
        ps.state_manager.set({
            {"background_opacity", "0"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "16"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        vps.push_back(ps);
    }
    for(int i = 0; i < 5; i++){
        string key = "ps" + to_string(i);
        cs.add_scene(&(vps[i]), key);
    }
    cs.inject_audio_and_render(SilenceSegment(25));
}
