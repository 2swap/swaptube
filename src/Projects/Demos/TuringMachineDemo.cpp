#include "../Scenes/Math/Beavers/TuringMachineScene.h"

void render_tm(const std::string& tc) {
    TuringMachineScene tms(tm_parse(tc));
    tms.manager.transition(MACRO, "iterations", "200", false);
    tms.manager.set("center_y", "<iterations> <slowdown> +");
    tms.manager.transition(MACRO, "zoom", "-2");
    stage_macroblock(SilenceBlock(16));
    tms.manager.set("slowdown", "0");
    tms.manager.transition(MICRO, "slowdown", "-10");
    tms.render_microblock();
    tms.render_microblock();
}
void render_video() {
    std::string tc1 = "1LB1RB_1LC0RC_0RA1RB";
    std::string tc2 = "1RB1LB_0RC0LA_1LC0LA";
    std::string bb5 = "1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA";
    std::string bb4 = "1RB1LB_1LA0LC_1RZ1LD_1RD0RA";
    //render_tm(tc1);
    render_tm(tc2);
    //render_tm(bb5);
    //render_tm(bb4);
}

