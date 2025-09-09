#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Math/RootFractalWithoutCudaScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video() {
    RootFractalScene rfs;
    CompositeScene cs;
    rfs.state_manager.set({{"terms", "18"}, {"dot_radius", "0.2"}});

    rfs.stage_macroblock(SilenceBlock(30), 7);
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", ".1"},
        {"coefficient0_i", ".1"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"center_x", "-.5"},
        {"center_y", ".5"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"zoom", "2"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "<t> 8 / sin"},
        {"coefficient0_i", "<t> 9 / sin"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "1"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"zoom", "4"},
    });
    rfs.render_microblock();
    rfs.state_manager.begin_timer("spin");
    rfs.state_manager.transition(MICRO, {
        {"center_x", "<spin> 5 / sin -1 *"},
        {"center_y", "<spin> 5 / cos"},
    });
    rfs.render_microblock();
}
