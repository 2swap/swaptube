#include "../Scenes/Math/ConwayScene.h"

void render_video() {
    ConwayScene cs;

    cs.manager.set("zoom", "-2");
    cs.manager.set("center_x", "0");
    cs.manager.set("ticks_opacity", "1");

    //cs.manager.transition(MICRO, "zoom", "-10");
    stage_macroblock(SilenceBlock(5), 3);
    while(remaining_microblocks_in_macroblock) {
        cs.render_microblock();
    }
}
