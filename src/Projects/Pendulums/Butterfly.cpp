#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    CompositeScene cs;
    vector<PendulumScene> vps;
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {2+i*.00001, 3.1415, .0, .0};
        PendulumScene ps(pendulum_state, 0xffff0000 + 50*i*256 + 50*(5-i));
        vps.push_back(ps);
    }
    for(int i = 0; i < 5; i++){
        string key = "ps" + to_string(i);
        cs.add_scene(&(vps[i]), key);
        cs.state_manager.add_equation(key + ".opacity", "0.4");
    }
    cs.inject_audio_and_render(SilenceSegment(9));
}
