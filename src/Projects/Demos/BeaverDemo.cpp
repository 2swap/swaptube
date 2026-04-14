#include "../Scenes/Math/Beavers/BeaverGridScene.h"

void bb52_progress() {
    BeaverGridScene bs(5, 2);
    bs.manager.set("zoom", "-16.3");
    bs.manager.set("sqrt_max_steps", "0");
    bs.manager.set("max_steps", "<sqrt_max_steps> 3 ^");
    bs.manager.set("center_x", "e <zoom> -1 * ^ {t} sin * 2 * 37900 -");
    bs.manager.set("center_y", "e <zoom> -1 * ^ {t} cos * 7350 +");
    stage_macroblock(SilenceBlock(50), 1);
    bs.manager.transition(MICRO, "sqrt_max_steps", "7.37");
    bs.manager.transition(MACRO, "zoom", "-3");
    bs.render_microblock();
}
void bb52() {
    BeaverGridScene bs(5, 2);
    bs.manager.set("zoom", "-16.3");
    bs.manager.set("max_steps", "400");
    bs.manager.set("center_x", "e <zoom> -1 * ^ {t} sin * 2 * 1034000 -");
    bs.manager.set("center_y", "e <zoom> -1 * ^ {t} cos * 7500 +");
    stage_macroblock(SilenceBlock(5), 7);
    bs.manager.transition(MACRO, "zoom", "-3");
    bs.render_microblock();
    bs.export_frame("bb52_1");
    bs.render_microblock();
    bs.export_frame("bb52_2");
    bs.render_microblock();
    bs.export_frame("bb52_3");
    bs.render_microblock();
    bs.export_frame("bb52_4");
    bs.render_microblock();
    bs.export_frame("bb52_5");
    bs.render_microblock();
    bs.export_frame("bb52_6");
    bs.render_microblock();
}
void bb52_norm() {
    BeaverGridScene bs(1, 2);
    bs.manager.set("max_steps", "400");
    bs.manager.set("center_x", "e <zoom> -1 * ^ {t} sin *");
    bs.manager.set("center_y", "e <zoom> -1 * ^ {t} cos *");
    bs.manager.set("num_states", "2");
    bs.manager.set("zoom", "1");
    stage_macroblock(SilenceBlock(3), 5);
    bs.manager.transition(MACRO, "zoom", "1");
    bs.render_microblock();
    bs.manager.set("num_states", "3");
    bs.render_microblock();
    bs.manager.set("num_states", "4");
    bs.render_microblock();
    bs.manager.set("num_states", "5");
    bs.render_microblock();
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
void bb33() {
    BeaverGridScene bs(3, 3);
    bs.manager.set("zoom", "-15");
    stage_macroblock(SilenceBlock(3), 3);
    bs.manager.transition(MICRO, "max_steps", "100");
    bs.render_microblock();
    bs.manager.transition(MICRO, "zoom", "-5");
    bs.render_microblock();
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
