#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    PendulumGridScene pgs(2.5, 0.1, 8, 0, 0);
    StateSet state = {
        {"physics_multiplier", "16"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / .1 *"},
    };
    pgs.state_manager.set(state);
    pgs.inject_audio_and_render(SilenceSegment(6));
    pgs.state_manager.add_microblock_transition("mode", "1");
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(SilenceSegment(2));
}
