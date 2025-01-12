#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    //SAVE_FRAME_PNGS = false;

    CompositeScene cs;
    PendulumGridScene pgs(2.49, 0.25, 25, 0, 0);
    PendulumScene ps({2.49, 0.25, 0, 0});
    StateSet state = {
        {"physics_multiplier", "16"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / .1 *"},
    };
    pgs.state_manager.set(state);
    ps.state_manager.set({
        {"background_opacity", "0"},
        {"pendulum_opacity", "1"},
        {"physics_multiplier", "16"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    });
    cs.add_scene(&pgs, "pgs");
    cs.add_scene(&ps, "ps");
    cs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.add_microblock_transition("mode", "1");
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio_and_render(SilenceSegment(2));
}
