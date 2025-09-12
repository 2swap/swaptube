#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video() {
    shared_ptr<RootFractalScene> fracs = make_shared<RootFractalScene>();
    CompositeScene cs;
    fracs->state_manager.set({{"terms", "15"}, {"zoom", ".3"}});

    fracs->stage_macroblock(SilenceBlock(40), 17);
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", ".1"},
        {"coefficient0_i", ".1"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"center_x", "-.5"},
        {"center_y", ".5"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "2"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<t> 8 / sin"},
        {"coefficient0_i", "<t> 9 / sin"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "1"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "7"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", ".3"},
    });
    fracs->render_microblock();
    fracs->state_manager.begin_timer("spin");
    fracs->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "0"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos"},
        {"coefficient0_i", "<spin> 2 / sin"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos 5 *"},
        {"coefficient0_i", "<spin> 2 / sin 5 *"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "1"},
        {"center_x", ".4"},
        {"center_y", ".4"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "2"},
        {"spin", to_string(fracs->state_manager.get_value("spin"))},
    });
    fracs->render_microblock();
    fracs->render_microblock();
}
