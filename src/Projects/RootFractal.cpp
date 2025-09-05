#include "../Scenes/Math/RootFractalScene.cpp"

void render_video() {
    RootFractalScene rfs;
    rfs.state_manager.set({{"terms", "11"}});

    rfs.stage_macroblock(SilenceBlock(1), 1);
    rfs.render_microblock();

    /*
    rfs.stage_macroblock(SilenceBlock(5), 1);
    rfs.state_manager.transition(MICRO, {
        {"zoom", "1"},
        {"center_y", "1"},
    });
    rfs.render_microblock();
    */

    /*
    rfs.stage_macroblock(SilenceBlock(10), 4);
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "0"},
        {"coefficient0_i", "1"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "2"},
        {"coefficient0_i", "0"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "2"},
        {"coefficient0_i", "-1"},
    });
    rfs.render_microblock();
    */

    rfs.stage_macroblock(SilenceBlock(10), 2);
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "<t> 2 / sin"},
        {"coefficient0_i", "<t> 3 / sin"},
    });
    rfs.render_microblock();
    rfs.render_microblock();
}
