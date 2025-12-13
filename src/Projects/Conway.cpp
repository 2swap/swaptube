#include "../Scenes/Math/ConwayScene.cpp"

void render_video() {
    ConwayScene cs;

    cs.manager.set("zoom", "-3");
    cs.manager.set("center_x", "20");
    cs.manager.set("ticks_opacity", "1");

    cs.manager.transition(MICRO, "zoom", "-10");
    cs.stage_macroblock(SilenceBlock(10), 1);
    cs.render_microblock();
}
