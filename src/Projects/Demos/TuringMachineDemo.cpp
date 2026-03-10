#include "../Scenes/Math/TuringMachineScene.h"

void render_tm(char* tc, int states, int symbols) {
    TuringMachine tm;
    parse_tm_from_string(tc, states, symbols, tm);
    TuringMachineScene tms(tm);
    tms.manager.transition(MACRO, "iterations", "200", false);
    tms.manager.set("center_y", "<iterations> <slowdown> +");
    tms.manager.transition(MACRO, "zoom", "-2");
    stage_macroblock(SilenceBlock(16), 2);
    tms.manager.set("slowdown", "0");
    tms.manager.transition(MICRO, "slowdown", "-10");
    tms.render_microblock();
    tms.render_microblock();
}
void render_video() {
    char tc1[21] = "1LB1RB_1LC0RC_0RA1RB";
    char tc2[21] = "1RB1LB_0RC0LA_1LC0LA";
    char bb5[35] = "1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA";
    char bb4[28] = "1RB1LB_1LA0LC_1RZ1LD_1RD0RA";
    render_tm(tc1, 3, 2);
    render_tm(tc2, 3, 2);
    render_tm(bb5, 5, 2);
    render_tm(bb4, 4, 2);
}

