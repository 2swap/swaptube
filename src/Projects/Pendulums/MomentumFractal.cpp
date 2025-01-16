#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    PendulumGridScene pgs(0, 0);
    pgs.state_manager.set({
        {"physics_multiplier", "32"},
        {"mode", "1"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 40 /"},
    });
    pgs.inject_audio_and_render(SilenceSegment(10));
    /*pgs.state_manager.microblock_transition({
        {"physics_multiplier", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"mode", "1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(SilenceSegment(2));
*/
}
