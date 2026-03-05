#include "../Scenes/Math/BeaverGridScene.h"

void bb52() {
    BeaverGridScene bs(5, 2);
    bs.manager.set("zoom", "-17");
    stage_macroblock(SilenceBlock(4), 4);
    bs.manager.transition(MACRO, "zoom", "-4");
    bs.manager.transition(MICRO, "max_steps", "400");
    bs.render_microblock();
    bs.manager.transition(MICRO, "center_x", "-40000");
    bs.render_microblock();
    bs.render_microblock();
    bs.render_microblock();
    stage_macroblock(SilenceBlock(4), 1);
    bs.manager.transition(MICRO, "center_x", "-40500");
    bs.manager.transition(MICRO, "center_y", "800");
    bs.render_microblock();
}
void bb42() {
    BeaverGridScene bs(4, 2);
    bs.manager.set("zoom", "-11");
    stage_macroblock(SilenceBlock(3), 2);
    bs.manager.transition(MICRO, "max_steps", "107");
    bs.render_microblock();
    bs.manager.transition(MICRO, "zoom", "-7");
    bs.render_microblock();
    stage_macroblock(SilenceBlock(3), 1);
    bs.manager.transition(MICRO, "center_x", "5000");
    bs.render_microblock();
    stage_macroblock(SilenceBlock(3), 1);
    bs.manager.transition(MICRO, "zoom", "-4");
    bs.render_microblock();
}
void bb32() {
    BeaverGridScene bs(3, 2);
    bs.manager.set("zoom", "-7");
    stage_macroblock(SilenceBlock(3), 2);
    bs.manager.transition(MICRO, "max_steps", "21");
    bs.render_microblock();
    bs.render_microblock();
}
void bb23() {
    BeaverGridScene bs(2, 3);
    bs.manager.set("zoom", "-10");
    stage_macroblock(SilenceBlock(1), 1);
    bs.manager.transition(MICRO, "zoom", "-5");
    bs.render_microblock();
    stage_macroblock(SilenceBlock(5), 2);
    bs.manager.transition(MICRO, "max_steps", "38");
    bs.render_microblock();
    bs.manager.transition(MICRO, "center_y", "1000");
    bs.render_microblock();
}

void render_video() {
    bb52();
}
