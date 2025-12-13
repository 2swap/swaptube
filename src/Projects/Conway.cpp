#include "../Scenes/Math/ConwayScene.cpp"

void render_video() {
    ConwayScene cs;

    cs.manager.set("zoom", "-9");
    cs.manager.set("center_x", "20");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.manager.transition(MICRO, "zoom", "-2");

    cs.stage_macroblock(SilenceBlock(2), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.manager.transition(MICRO, "center_x", "100");

    cs.stage_macroblock(SilenceBlock(2), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.manager.transition(MICRO, "center_y", "100");

    cs.stage_macroblock(SilenceBlock(2), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.manager.transition(MICRO, "center_x", "0");
    cs.manager.transition(MICRO, "center_y", "0");
    cs.manager.transition(MICRO, "zoom", "-9");

    cs.stage_macroblock(SilenceBlock(2), 2);
    cs.render_microblock();
    cs.render_microblock();
}
