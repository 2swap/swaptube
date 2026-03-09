#include "../Scenes/Math/TuringMachineScene.h"

void render_video() {
    TuringMachine tm;
    char bb4[28] = "1RB1LB_1LA0LC_1RZ1LD_1RD0RA";
    parse_tm_from_string(bb4, 4, 2, tm);
    TuringMachineScene tms(tm);
    tms.manager.transition(MICRO, "iterations", "40");
    stage_macroblock(SilenceBlock(2), 1);
    tms.render_microblock();
}
