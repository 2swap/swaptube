#include "../Scenes/Math/ConwayScene.cpp"

void render_video() {
    ConwayScene cs;

    cs.manager.set("zoom", "-2");
    cs.manager.set("center_x", "0");
    cs.manager.set("ticks_opacity", "1");

    //cs.manager.transition(MICRO, "zoom", "-10");
    cs.stage_macroblock(SilenceBlock(3), 1);
    cs.render_microblock();
}
