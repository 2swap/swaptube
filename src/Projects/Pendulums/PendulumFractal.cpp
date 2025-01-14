#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    PendulumGridScene pgs(0, 0);
    pgs.state_manager.set({
        {"physics_multiplier", "16"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / .1 *"},
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
        {"zoom", "1 6.28 /"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.28 /"},
        {"physics_multiplier", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 24 /"},
        {"physics_multiplier", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"mode", "1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(SilenceSegment(2));
}
