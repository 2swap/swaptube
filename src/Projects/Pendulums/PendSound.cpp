#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    PendulumState pendulum_state = {2, .3, .0, .0};
    PendulumScene ps(pendulum_state, 0xffffffff);
    vector<float> left;
    vector<float> right;
    ps.generate_audio(8, left, right);
    ps.inject_audio_and_render(GeneratedSegment(left, right));
}
