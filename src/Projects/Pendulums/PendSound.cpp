#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;

    vector<PendulumScene> ps;
    ps.push_back(PendulumScene({0.3, .3, .0, .0}));
    ps.push_back(PendulumScene({3.6, 3, .0, .0}));
    ps.push_back(PendulumScene({2.49, .25, .0, .0}));
    ps.push_back(PendulumScene({2, .3, .0, .0}));
    ps.push_back(PendulumScene({0.1, .3, .0, .0}));
    for(int i = 0; i < ps.size(); i++){
        vector<float> left;
        vector<float> right;
        ps[i].state_manager.set({
            {"physics_multiplier", "1"},
            {"pendulum_opacity", "1"},
            {"background_opacity", "0.1"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        ps[i].generate_audio(4, left, right);
        ps[i].inject_audio_and_render(GeneratedSegment(left, right));
    }
}
