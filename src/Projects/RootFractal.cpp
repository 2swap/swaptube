#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video() {
    RootFractalScene rfs;
    CompositeScene cs;
    rfs.state_manager.set({{"terms", "15"}, {"zoom", ".3"}});

    rfs.stage_macroblock(SilenceBlock(40), 17);
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
        {"zoom", "7"},
    });
    rfs.render_microblock();
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"zoom", ".3"},
    });
    rfs.render_microblock();
    rfs.state_manager.begin_timer("spin");
    rfs.state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "0"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos"},
        {"coefficient0_i", "<spin> 2 / sin"},
    });
    rfs.render_microblock();
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos 5 *"},
        {"coefficient0_i", "<spin> 2 / sin 5 *"},
    });
    rfs.render_microblock();
    rfs.render_microblock();
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"zoom", "1"},
        {"center_x", ".4"},
        {"center_y", ".4"},
    });
    rfs.render_microblock();
    rfs.state_manager.transition(MICRO, {
        {"zoom", "2"},
        {"spin", to_string(rfs.state_manager.get_value("spin"))},
    });
    rfs.render_microblock();
    rfs.render_microblock();
}
