#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    CompositeScene cs;
    vector<PendulumScene> vps;
    for(int i = 0; i < 10; i++){
        PendulumState pendulum_state = {2, 3.1415, .0, .0};
        PendulumScene ps(pendulum_state, 0xffff0000 + 20*i*256 + 255-20*i);
        StateSet state = {
            {"physics_multiplier", to_string(i+1)},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        };
        ps.state_manager.set(state);
        vps.push_back(ps);
    }
    for(int i = 0; i < vps.size(); i++){
        string key = "ps" + to_string(i);
        cs.add_scene(&(vps[i]), key);
        cs.state_manager.add_equation(key + ".opacity", "0.3");
    }
    cs.inject_audio_and_render(SilenceSegment(20));
}
