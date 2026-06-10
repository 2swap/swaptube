#include "../Scenes/Math/ConwayScene.h"

void render_video() {
    ConwayScene cs(ivec2(20000,20000));

    cs.manager.set("zoom", "-3");
    cs.manager.set("center_x", "18700 {t} .2 * cos 200 * +");
    cs.manager.set("center_y", "-9000 {t} .2 * sin 200 * +");
    cs.manager.set("ticks_opacity", "1");

    stage_macroblock(SilenceBlock(12), 8);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-11");
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-3");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
}
