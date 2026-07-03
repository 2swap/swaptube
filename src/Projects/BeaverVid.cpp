#include "../Scenes/Common/CompositeScene.h"
//#include "../Scenes/Common/PauseScene.h"
#include "../Scenes/Common/TwoswapScene.h"

#include "../Scenes/Media/LatexScene.h"
//#include "../Scenes/Media/PngScene.h"
//#include "../Scenes/Media/Mp4Scene.h"
//#include "../Scenes/Media/WhitePaperScene.h"

#include "../Scenes/Math/Beavers/TuringMachineScene.h"
#include "../Scenes/Math/Beavers/BeaverGridScene.h"
#include "../Scenes/Math/Beavers/BeaverGridTNFScene.h"
#include "../Scenes/Math/Beavers/BeaverGridTNF3DScene.h"
#include "../Scenes/Math/Beavers/BeaverTNF3DScene.h"


struct Path {
    int pathlen = 0;
    int action[CODON_MEM_LIMIT];
};

void core_child(vec3& lower, vec3& upper, int state, int symb, int dir, int states, int symbs, float shell_border, float core_border) {
    vec3 border = (upper - lower) * core_border;
    lower += border;
    upper -= border;
    vec3 shell_size = (upper - lower) / vec3(states, symbs, 2);
    lower += vec3(state, symb, dir) * shell_size;
    upper = lower + shell_size;
    border = (upper - lower) * shell_border;
    lower += border;
    upper -= border;
}

vec3 target_tm(TuringMachine& tm, Path& p, float shell_border, float core_border) {
    vec3 lower = vec3(0);
    vec3 upper = vec3(1);
    int num_states = 2;
    int num_symbols = 2;
    for (int i = 0; i < p.pathlen; i++) {
        core_child(lower, upper, tm.next_state[p.action[i]], tm.write_symbol[p.action[i]], tm.left_right[p.action[i]], num_states, num_symbols, shell_border, core_border);
	num_states += (int)(tm.next_state[p.action[i]] == num_states - 1);
	num_symbols += (int)(tm.write_symbol[p.action[i]] == num_symbols - 1);
    }
    return 0.5 * (lower + upper);
}



void set_transition(TuringMachine& tm, int state, int symbol, int ws, bool lr, int ns) {
    int action_layer = max(state, symbol) - 1;
    int action_side = (int)(state < symbol);
    int action_index = action_layer * action_layer + 2 * (state + symbol) + action_side - 1;
    if (action_index < CODON_MEM_LIMIT) {
        tm.write_symbol[action_index] = ws;
        tm.left_right[action_index] = lr;
        tm.next_state[action_index] = ns;
    }
}

void parse_tmstring(char* s, int num_states, int num_symbols, TuringMachine& tm) {
    tm.num_symbols = num_symbols;
    tm.num_states = num_states;
    for(int state = 0; state < num_states; state++) {
        for(int symbol = 0; symbol < num_symbols; symbol++) {
            int string_index = state * (num_symbols * 3 + 1) + symbol * 3;
            char ns = s[string_index+2];
            set_transition(tm, state, symbol, s[string_index] - '0', s[string_index+1] == 'R', ns == '-' ? -1 : ns - 'A');
        }
    }
}



void TNF3Dtest() {
    BeaverTNF3DScene bs;
    bs.manager.set("max_steps", "50");
    bs.manager.set("core_border", "0");
    bs.manager.set("shell_border", "0");

    char bigfoot[30] = "1RB2RA1LC_2LC1RB2RB_1R-2LA1LA";
    TuringMachine tm;
    Path pbf = {8, {0, 1, 6, 7, 8, 5, 3, 2}};
    parse_tmstring(bigfoot, 3, 3, tm);
    vec3 center = target_tm(tm, pbf, 0, 0);
    printf("\n(%f,%f,%f)", center.x, center.y, center.z);
    // 0.580285,  0.992403,  0.649414

    quat camera = get_quat(vec3(0, 0, -1), vec3(0, 1, 0));
    printf("\n%f+%fi+%fj+%fk\n", camera.u, camera.i, camera.j, camera.k);
    bs.manager.set("q1", std::to_string(camera.u));
    bs.manager.set("qi", std::to_string(camera.i));
    bs.manager.set("qj", std::to_string(camera.j));
    bs.manager.set("qk", std::to_string(camera.k));
    //bs.manager.set("target_x", /*"0.593"*/ "0.616");
    //bs.manager.set("target_y", "0.4");
    //bs.manager.set("target_z", /*"0.61"*/ "0.59");
    bs.manager.set("target_x", std::to_string(center.x));
    bs.manager.set("target_y", std::to_string(center.y));
    bs.manager.set("target_z", std::to_string(center.z));
    bs.manager.set("zoom", "0.3");
    bs.manager.set("camera_distance", "e <zoom> -1 * ^");
    //bs.manager.set("ancestor_offset", "<zoom> 1.5 * 2 -");
    bs.manager.set("scale_x", "e 9 0.55 * ^");
    bs.manager.set("scale_y", "e 9 0.55 * ^");
    bs.manager.set("scale_z", "1");
    bs.manager.set("brightness_offset", "<zoom> -0.06 *");
    //bs.manager.set("brightness_offset", "0.3 <zoom> 3 ^ 125 / -");
    stage_macroblock(SilenceBlock(6), 1);
    bs.manager.transition(MICRO, "zoom", "1");
    //bs.manager.transition(MICRO, "q1", "0.15");
    //bs.manager.set("q1", "0.15");
    //bs.manager.transition(MICRO, "target_z", "0.605");
    bs.render_microblock();
}





/*
###  ###   #### ##### #   # ##### ###    ###
#  # #  #  #      #   ##  #   #   #  #  #   #
###  ###   ###    #   # # #   #   ###   #   #
#    #  #  #      #   #  ##   #   #  #  #   #
#    #   # #### ##### #   #   #   #   #  ###
*/

void preintro(CompositeScene& cs) {
    /*
    "You are looking at every possible computer program." (super deep zoom OUT of the 3d TNF grid, at the end of which the space-time diagram appears on the side, showing the empty TM)
    "Most of them just loop forever," (zoom in on a TC while the space-time diagram evolves appropriately)
    "or bounce from side to side." (zoom out and in on a bouncer)
    "Some count to infinity, usually in binary" (zoom in on a binary counter)
    "But occasionally, you might stumble upon Cryptids, the lovecraftian machines that stand firmly between us and a complete understanding of mathematics"
    (zoom in on hydra, while spooky visual effects creep in from the edges and spooky audio effects fade in as well and the 3d TNF grid "glitches out" with hydra blinking into and out of existence)
    */
    shared_ptr<BeaverGridTNF3DScene> tnfs = make_shared<BeaverGridTNF3DScene>();
    shared_ptr<TuringMachineScene> tms;
    TuringMachine tm;
    char bb4[28] = "1RB1LB_1LA0LC_1RZ1LD_1RD0RA";
    char tc[14] = "0RB1RA_1LA1RB";
    char bouncer[14] = "0RB1LA_1LA1RB";
    char counter[14] = "0RB0LA_1LA1RB";
    char hydra[32] = "1RB3RB1RZ3LA1RA_2LA3RA4LB0LB0LA";

    cs.add_scene(tnfs, "tnfs");
    parse_tmstring(bb4, 4, 2, tm);
    Path p = {7, {0, 1, 2, 3, 6, 9, 11}};
    vec3 center = target_tm(tm, p, 0.1, 0.1);

    parse_tmstring(tc, 2, 2, tm);

    tnfs->manager.set({
        {"spin_offset", "-14"},
        {"q1", "{t} <spin_offset> + 10 / sin"},
        {"qi", "0"},
        {"qj", "{t} <spin_offset> + 10 / cos -1 *"},
        {"qk", "0"},
        {"target_x", std::to_string(center.x) + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_y", std::to_string(center.y) + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_z", std::to_string(center.z) + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"max_steps", "200"},
        {"zoom", "7.35"},
        {"camera_distance", "e <zoom> -1 * ^"},
        {"ancestor_offset", "<zoom> 1.5 * 2 -"},
        {"scale_x", "e <zoom> 0.55 * ^"},
        {"scale_y", "e <zoom> 0.33 * ^"},
        {"scale_z", "1"}
    });
    stage_macroblock(SilenceBlock(5), 1);

    tnfs->manager.transition(MICRO, "zoom", "0");
    cs.render_microblock();


    p.action[2] = 3;
    p.action[3] = 2;
    p.pathlen = 3;
    vec3 center_2x2s = target_tm(tm, p, 0.1, 0.1);
    std::string center_2x2s_x = std::to_string(center_2x2s.x);
    std::string center_2x2s_y = std::to_string(center_2x2s.y);
    std::string center_2x2s_z = std::to_string(center_2x2s.z);
    p.pathlen = 4;
    center = target_tm(tm, p, 0.1, 0.1);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 4.5 - 60 / -"},
        {"iterations", "0"}
    });
    stage_macroblock(SilenceBlock(6), 3);

    cs.manager.set("tms.x", "1");
    cs.manager.transition(MICRO, "tms.x", "0.65");
    cs.manager.transition(MICRO, "tnfs.x", "0.3");
    tnfs->manager.transition(MICRO, "w", "0.6");
    //tnfs->manager.transition(MICRO, "center_x", "0.375");
    tnfs->manager.set({
        {"target_x", "0.5"},
        {"target_y", "0.5"},
        {"target_z", "0.5"},
        {"highlight_x", std::to_string(center.x)},
        {"highlight_y", std::to_string(center.y)},
        {"highlight_z", std::to_string(center.z)},
        {"scale_x", "1"},
        {"scale_y", "1"}
    });
    tnfs->manager.transition(MICRO, {
        {"target_x", center_2x2s_x},
        {"target_y", center_2x2s_y},
        {"target_z", center_2x2s_z},
        {"zoom", "0.6"}
    });
    cs.render_microblock();

    tms->manager.transition(MICRO, "iterations", "50");
    tnfs->manager.transition(MICRO, {
        {"target_x", std::to_string(center.x)},
        {"target_y", std::to_string(center.y)},
        {"target_z", std::to_string(center.z)},
        {"spin_offset", "-8 {t} 1.5 / -"},
        {"zoom", "4.5"},
        {"scale_x", "1.8"},
        {"scale_y", "1.4"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();

    tms->manager.transition(MICRO, "iterations", "4.5");
    tnfs->manager.transition(MICRO, {
        {"target_x", center_2x2s_x},
        {"target_y", center_2x2s_y},
        {"target_z", center_2x2s_z},
        {"zoom", "3.1"},
        {"highlight_intensity", "0"}
    });
    cs.render_microblock();

    cs.remove_subscene("tms");


    parse_tmstring(bouncer, 2, 2, tm);
    center = target_tm(tm, p, 0.1, 0.1);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    cs.manager.set("tms.x", "0.65");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 4.5 - 200 / -"},
        {"iterations", "4.5"}
    });
    stage_macroblock(SilenceBlock(4), 2);

    tms->manager.transition(MICRO, "iterations", "50");
    tnfs->manager.set({
        {"highlight_x", std::to_string(center.x)},
        {"highlight_y", std::to_string(center.y)},
        {"highlight_z", std::to_string(center.z)},
    });
    tnfs->manager.transition(MICRO, {
        {"target_x", std::to_string(center.x)},
        {"target_y", std::to_string(center.y)},
        {"target_z", std::to_string(center.z)},
        {"zoom", "4.2"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();

    tms->manager.transition(MICRO, "iterations", "4.5");
    tnfs->manager.transition(MICRO, {
        {"target_x", center_2x2s_x},
        {"target_y", center_2x2s_y},
        {"target_z", center_2x2s_z},
        {"zoom", "3.1"},
        {"highlight_intensity", "0"}
    });
    cs.render_microblock();

    cs.remove_subscene("tms");


    parse_tmstring(counter, 2, 2, tm);
    center = target_tm(tm, p, 0.1, 0.1);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    cs.manager.set("tms.x", "0.65");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 4.5 - 400 / -"},
        {"iterations", "4.5"}
    });
    stage_macroblock(SilenceBlock(8), 2);

    tms->manager.transition(MICRO, "iterations", "260");
    tnfs->manager.set({
        {"highlight_x", std::to_string(center.x)},
        {"highlight_y", std::to_string(center.y)},
        {"highlight_z", std::to_string(center.z)},
    });
    tnfs->manager.transition(MICRO, {
        {"target_x", std::to_string(center.x)},
        {"target_y", std::to_string(center.y)},
        {"target_z", std::to_string(center.z)},
        {"zoom", "4.45"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();

    cs.manager.transition(MICRO, "tms.x", "0.9");
    tms->manager.transition(MICRO, {
        {"iterations", "0"},
        {"zoom", "-1.5 <iterations> -400 / -"}
    });
    tnfs->manager.set({
        {"target_x", std::to_string(center.x) + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_y", std::to_string(center.y) + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_z", std::to_string(center.z) + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"}
    });
    tnfs->manager.transition(MICRO, {
        {"spin_offset", "8 {t} 1.5 * -"},
        {"zoom", "0"},
        {"highlight_intensity", "0"},
        {"scale_x", "1"},
        {"scale_y", "1"}
    });
    cs.render_microblock();

    cs.remove_subscene("tms");


    Path phydra = {9, {0, 1, 2, 7, 12, 3, 17, 19, 10}};
    parse_tmstring(hydra, 2, 5, tm);
    center = target_tm(tm, phydra, 0.1, 0.1);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    cs.manager.set("tms.x", "0.9");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 700 / -"},
        {"iterations", "0"}
    });
    stage_macroblock(SilenceBlock(8), 1);

    tms->manager.transition(MICRO, {
        {"iterations", "600"},
    });
    tnfs->manager.set({
        {"highlight_x", std::to_string(center.x)},
        {"highlight_y", std::to_string(center.y)},
        {"highlight_z", std::to_string(center.z)},
        {"target_x", std::to_string(center.x) + " 0.5 - e 1 1 1 {microblock_fraction} - 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_y", std::to_string(center.y) + " 0.5 - e 1 1 1 {microblock_fraction} - 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_z", std::to_string(center.z) + " 0.5 - e 1 1 1 {microblock_fraction} - 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"scale_x", "e <zoom> 0.33 * ^"},
        {"scale_y", "e <zoom> 0.77 * ^"}
    });
    tnfs->manager.transition(MICRO, {
        {"zoom", "10"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();
}

/*
##### #   # ##### ###    ###
  #   ##  #   #   #  #  #   #
  #   # # #   #   ###   #   #
  #   #  ##   #   #  #  #   #
##### #   #   #   #   #  ###
*/

//void intro() {
//}





void render_video() {
    CompositeScene cs;
    TNF3Dtest();
    //preintro(cs);
}
