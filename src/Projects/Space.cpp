#include "../Scenes/Math/GeodesicScene.cpp"

void render_video() {
    GeodesicScene gs;

    gs.state.set("fov", "2");
    gs.state.set("w_eq", "(x) (x) * (y) (y) * (z) (z) * + +");

    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.state.transition(MICRO, {
        {"qj", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.state.transition(MICRO, {
        {"qi", "{t} .5 * sin .1 *"},
        {"qj", "{t} .3 * cos .1 *"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.state.transition(MICRO, {
        {"z", ".75"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.state.transition(MICRO, {
        {"z", "-.75"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.state.transition(MICRO, {
        {"z", "0"},
        {"x", "-.75"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.state.transition(MICRO, {
        {"z", "0"},
        {"x", ".75"},
    });
    gs.render_microblock();
    gs.render_microblock();
}
