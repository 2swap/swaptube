#include "../Scenes/Math/GeodesicScene.cpp"

/*
void render_video_paraboloid() {
    GeodesicScene gs;

    gs.manager.set("fov", "2");
    gs.manager.set("z", "5");
    gs.manager.set("w_eq", "(x) (x) * (y) (y) * (z) (z) * + +");
    //gs.manager.set("intensity", "2 {t} 2 * sin ^ 5 *");

    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"q1", ".5"},
        {"qj", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", "{t} .5 * sin .1 *"},
        {"qj", "{t} .3 * cos .1 *"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"z", "5"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"z", "-5"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"x", "-5"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"x", "5"},
    });
    gs.render_microblock();
    gs.render_microblock();
}
*/

void render_video() {
    GeodesicScene gs;

    gs.manager.set("fov", "2");
    gs.manager.set("z", "5");

    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"q1", ".5"},
        {"qj", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", "{t} .5 * sin .1 *"},
        {"qj", "{t} .3 * cos .1 *"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"z", "3"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"z", "7"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"x", "-2"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(2), 2);
    gs.manager.transition(MICRO, {
        {"x", "2"},
    });
    gs.render_microblock();
    gs.render_microblock();
}
