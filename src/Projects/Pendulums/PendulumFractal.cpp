#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    PendulumGridScene pgs(0, 0);
    StateSet state = {
        {"physics_multiplier", "4"},
        {"rk4_step_size",      "1 30 / .5 *"},
    };
    pgs.state_manager.set(state);
    pgs.inject_audio_and_render(SilenceSegment(10));
}
