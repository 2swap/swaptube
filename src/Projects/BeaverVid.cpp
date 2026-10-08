#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Common/AffineScene.h"
//#include "../Scenes/Common/PauseScene.h"
//#include "../Scenes/Common/TwoswapScene.h"

#include "../Scenes/Media/LatexScene.h"
//#include "../Scenes/Media/PngScene.h"
//#include "../Scenes/Media/Mp4Scene.h"
//#include "../Scenes/Media/WhitePaperScene.h"

#include "../Scenes/Math/Beavers/TuringMachineScene.h"
//#include "../Scenes/Math/Beavers/BeaverGridTNFScene.h"
//#include "../Scenes/Math/Beavers/BeaverGridTNF3DScene.h"
#include "../Scenes/Math/Beavers/BeaverGridSpacetimeScene.h"
#include "../Scenes/Math/Beavers/BeaverTNF3DScene.h"
#include "../Scenes/Math/Beavers/BeaverIndividualScene.h"

#include "../IO/Latex.h"
#include "../IO/PNG.h"
#include <vector>
#include <string>

extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);


struct Path {
    int pathlen = 0;
    int action[CODON_MEM_LIMIT];
};

string strmul(string s0, int n) {
    string s = "";
    for (int i=0; i<n; i++) {
        s += s0;
    }
    return s;
}

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

ivec4 icons_whnm;
uint32_t* icons;
int icons_len;

uint32_t* init_icons(std::vector<std::string> pngnames, ivec2 wh) {
    int icon_size = wh.x * wh.y * sizeof(uint32_t);
    int icon_amount = pngnames.size();
    uint32_t* d_icons = cuda_alloc_pixels_on_device(icon_amount * icon_size);
    uint32_t* h_icons[icon_amount];
    for (int i=0; i<icon_amount; i++) h_icons[i] = new uint32_t[wh.x * wh.y];
    Pixels pix;
    std::vector<Pixels> scaled;
    scaled.resize(icon_amount);
    for (int i=0; i<icon_amount; i++) {
        png_to_pix(pix, pngnames[i]);
        pix.scale_to_bounding_box(wh.x, wh.y, scaled[i]);
        for (int j=0; j<wh.x*wh.y; j++) {
            h_icons[i][j] = scaled[i].pixels[j];
        }
    }
    for (int i=0; i<icon_amount; i++) {
        cuda_copy_pixels_to_device(h_icons[i], icon_size, d_icons + i * wh.x * wh.y);
    }
    return d_icons;
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

void parse_tmstring(string s, int num_states, int num_symbols, TuringMachine& tm) {
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



void begin_zoom(Scene& s, string zoom_var, string x_var, string y_var, float target_zoom, float target_x, float target_y, string progress_expr="0 1 {microblock_fraction} smoothlerp") {
    float zoom = s.manager.get_local_value(zoom_var);
    float x = s.manager.get_local_value(x_var);
    float y = s.manager.get_local_value(y_var);
    float e = 2.718281828f;
    if (zoom == target_zoom) {
        s.manager.set({
            {zoom_var, std::to_string(zoom)},
            {x_var, std::to_string(x) + " " + std::to_string(target_x) + " " + progress_expr + " lerp"},
            {y_var, std::to_string(y) + " " + std::to_string(target_y) + " " + progress_expr + " lerp"},
        });
    }
    else {
        vec2 unit = vec2(x-target_x, y-target_y) / (pow(e,-zoom) - pow(e,-target_zoom));
        vec2 center = vec2(target_x,target_y) - unit*pow(e,-target_zoom);
        s.manager.set({
            {zoom_var, std::to_string(zoom) + " " + std::to_string(target_zoom) + " " + progress_expr + " lerp"},
            {x_var, std::to_string(center.x) + " " + std::to_string(unit.x) + " e <" + zoom_var + "> -1 * ^ * +"},
            {y_var, std::to_string(center.y) + " " + std::to_string(unit.y) + " e <" + zoom_var + "> -1 * ^ * +"},
        });
    }
}



void add_point_spacetime(shared_ptr<AffineScene>& as, string tms_name) {
    int id = as->num_points;
    as->add_point();
    as->manager.set({
        {"p" + std::to_string(id) + "xeq", "<p" + std::to_string(id) + "x> [" + tms_name + ".stfx] - 0.5625 e [" + tms_name + ".zoom] ^ * * [" + tms_name + ".x] +"},
        {"p" + std::to_string(id) + "yeq", "<p" + std::to_string(id) + "y> [" + tms_name + ".stfy] - [" + tms_name + ".vs] e [" + tms_name + ".zoom] ^ * * [" + tms_name + ".y] +"},
    });
}

void add_coin(shared_ptr<AffineScene>& as, int center, vector<string> sides={"H", "T"}, vector<uint32_t> colors={0xff60a0ff, 0xfff0ff40}) {
    int tn = as->text_anchors.size();
    as->add_text(center, {latex_color(colors[0], sides[0])});
    as->add_text(center, {latex_color(colors[1], sides[1])});
    as->manager.set({
        //{"p" + std::to_string(center) + "yeq", as->manager.get_equation_string("p" + std::to_string(center) + "yeq") + " <c" + std::to_string(center) + "t> <c" + std::to_string(center) + "a> sin 1 2 <p" + std::to_string(center) + "m11> 0 > * - * * +"},
        {"p" + std::to_string(center) + "o", "<c" + std::to_string(center) + "o>"},
        {"p" + std::to_string(center) + "t", "<c" + std::to_string(center) + "r>"},
        {"p" + std::to_string(center) + "m11", "1 <c" + std::to_string(center) + "a> cos /"},
        {"p" + std::to_string(center) + "r", std::to_string(((colors[0] >> 16) & 0x000000ff) / 256.0f) + " " + std::to_string(((colors[1] >> 16) & 0x000000ff) / 256.0f) + " <p" + std::to_string(center) + "m11> 0 < lerp"},
        {"p" + std::to_string(center) + "g", std::to_string(((colors[0] >> 8) & 0x000000ff) / 256.0f) + " " + std::to_string(((colors[1] >> 8) & 0x000000ff) / 256.0f) + " <p" + std::to_string(center) + "m11> 0 < lerp"},
        {"p" + std::to_string(center) + "b", std::to_string((colors[0] & 0x000000ff) / 256.0f) + " " + std::to_string((colors[1] & 0x000000ff) / 256.0f) + " <p" + std::to_string(center) + "m11> 0 < lerp"},
        {"p" + std::to_string(center) + "s", "2"},
        {"t" + std::to_string(tn  ) + "o", "<c" + std::to_string(center) + "o>"},
        {"t" + std::to_string(tn  ) + "h", "<c" + std::to_string(center) + "r> 2 *"},
        {"t" + std::to_string(tn  ) + "m11", "1 <c" + std::to_string(center) + "a> cos 0 max /"},
        {"t" + std::to_string(tn+1) + "o", "<c" + std::to_string(center) + "o>"},
        {"t" + std::to_string(tn+1) + "h", "<c" + std::to_string(center) + "r> 2 *"},
        {"t" + std::to_string(tn+1) + "m11", "1 <c" + std::to_string(center) + "a> cos -1 * 0 max /"},

        {"c" + std::to_string(center) + "o", "1"},
        {"c" + std::to_string(center) + "r", "0.06"},
        {"c" + std::to_string(center) + "a", "0"},
    });
}

void add_rect(shared_ptr<AffineScene>& as, int center, int corner, string label="", int tms_link=-1) {
    int pn = as->num_points;
    int ln = as->line_endpoints.size();
    int tn = as->text_anchors.size();
    as->add_point({center, corner});
    as->add_point();
    as->add_line(pn+1, corner);
    as->add_line(pn+1, corner);
    as->add_line(pn+1, pn);
    as->add_line(pn+1, pn);
    as->add_text(center, {label});
    as->manager.set({
        {"p" + std::to_string(center) + "o", "0"},
        {"p" + std::to_string(corner) + "o", "0"},
        {"p" + std::to_string(pn) + "w" + std::to_string(corner), "-0.5"},
        {"p" + std::to_string(pn) + "o", "0"},
        {"p" + std::to_string(pn+1) + "x", "<p" + std::to_string(corner) + "xeq>"},
        {"p" + std::to_string(pn+1) + "y", "<p" + std::to_string(pn) + "yeq>"},
        {"p" + std::to_string(pn+1) + "o", "0"},
        {"l" + std::to_string(ln+1) + "x", "<p" + std::to_string(pn) + "xeq> <p" + std::to_string(corner) + "xeq> -"},
        {"l" + std::to_string(ln+3) + "y", "<p" + std::to_string(corner) + "yeq> <p" + std::to_string(pn) + "yeq> -"},
        {"t" + std::to_string(tn) + "yabs", "<p" + std::to_string(corner) + "yeq> <p" + std::to_string(pn) + "yeq> - abs <t" + std::to_string(tn) + "h> + 2 /"},
        {"t" + std::to_string(tn) + "y", "<t" + std::to_string(tn) + "yabs>"},

        {"r" + std::to_string(center) + "o", "1"},
        {"l" + std::to_string(ln  ) + "o", "<r" + std::to_string(center) + "o>"},
        {"l" + std::to_string(ln+1) + "o", "<r" + std::to_string(center) + "o>"},
        {"l" + std::to_string(ln+2) + "o", "<r" + std::to_string(center) + "o>"},
        {"l" + std::to_string(ln+3) + "o", "<r" + std::to_string(center) + "o>"},
        {"t" + std::to_string(tn  ) + "o", "<r" + std::to_string(center) + "o>"},
    });
    if (tms_link != -1) as->manager.set({
        {"r" + std::to_string(center) + "o", "[tms" + std::to_string(tms_link) + ".opacity]"},
        {"p" + std::to_string(center) + "x", "[tms" + std::to_string(tms_link) + ".x]"},
        {"p" + std::to_string(center) + "y", "[tms" + std::to_string(tms_link) + ".y]"},
        {"p" + std::to_string(corner) + "x", "[tms" + std::to_string(tms_link) + ".w] 2 /"},
        {"p" + std::to_string(corner) + "y", "[tms" + std::to_string(tms_link) + ".h] 2 /"},
    });
}

void create_spectrum(shared_ptr<AffineScene>& as, bool transition=false) {
    int pn = as->num_points;
    int ln = as->line_endpoints.size();
    as->add_point();
    as->add_point();
    as->add_point({pn,pn+1});
    as->add_line(pn,pn+2);
    as->add_line(pn+1,pn+2);
    as->manager.set({
        {"p" + std::to_string(pn) + "x", std::to_string(0.25*(1+transition))},
        {"p" + std::to_string(pn) + "y", "0.5"},
        {"p" + std::to_string(pn) + "o", "0"},
        {"p" + std::to_string(pn+1) + "x", "1 <p" + std::to_string(pn) + "x> -"},
        {"p" + std::to_string(pn+1) + "y", "<p" + std::to_string(pn) + "y>"},
        {"p" + std::to_string(pn+1) + "o", "0"},
        {"p" + std::to_string(pn+2) + "o", "0"},
        {"l" + std::to_string(ln) + "r1", std::to_string((int)(transition))},
        {"l" + std::to_string(ln) + "g1", std::to_string((int)(transition))},
        {"l" + std::to_string(ln+1) + "g1", std::to_string((int)(transition))},
        {"l" + std::to_string(ln+1) + "b1", std::to_string((int)(transition))},
    });
    if (transition) {
        as->manager.transition(MICRO, {
            {"p" + std::to_string(pn) + "x", "0.25"},
            {"l" + std::to_string(ln) + "r1", "0"},
            {"l" + std::to_string(ln) + "g1", "0"},
            {"l" + std::to_string(ln+1) + "g1", "0"},
            {"l" + std::to_string(ln+1) + "b1", "0"},
        });
    }
}

void connect_to_spectrum(shared_ptr<AffineScene>& as, int spect_start, int rect_start, float conn_src, float conn_dst=0.5, bool transition=true) {
    int pn = as->num_points;
    int ln = as->line_endpoints.size();
    as->add_point({spect_start, spect_start+1});
    as->add_point({pn});
    as->add_line(pn, pn+1);
    as->draw_order.insert(as->draw_order.begin(), {LINE, ln});
    as->draw_order.pop_back();
    as->manager.set({
        {"p" + std::to_string(pn) + "w" + std::to_string(spect_start), std::to_string(1-conn_src)},
        {"p" + std::to_string(pn) + "w" + std::to_string(spect_start+1), std::to_string(conn_src)},
        {"p" + std::to_string(pn) + "o", "0"},
        {"p" + std::to_string(pn+1) + "o", "0"},
        {"l" + std::to_string(ln) + "t", "0.002"},
        {"l" + std::to_string(ln) + "r1", "0.4"},
        {"l" + std::to_string(ln) + "g1", "0.4"},
        {"l" + std::to_string(ln) + "b1", "0.4"},
        {"l" + std::to_string(ln) + "r2", "0.4"},
        {"l" + std::to_string(ln) + "g2", "0.4"},
        {"l" + std::to_string(ln) + "b2", "0.4"},
        {"l" + std::to_string(ln) + "o", "<r" + std::to_string(rect_start) + "o> 1 <p" + std::to_string(rect_start) + "yeq> <p" + std::to_string(rect_start+1) + "y> abs + <p" + std::to_string(pn) + "yeq> > <p" + std::to_string(rect_start) + "yeq> <p" + std::to_string(rect_start+1) + "y> abs - <p" + std::to_string(pn) + "yeq> < * - *"},
    });
    string xeq = "<p" + std::to_string(rect_start) + "xeq> <p" + std::to_string(rect_start+1) + "x> abs 2 " + std::to_string(conn_dst) + " * 1 - * +";
    string yeq = "<p" + std::to_string(rect_start) + "yeq> <p" + std::to_string(rect_start+1) + "y> abs 1 2 <p" + std::to_string(rect_start) + "yeq> <p" + std::to_string(pn) + "yeq> > * - * +";
    if (transition) as->manager.transition(MICRO, {{"p" + std::to_string(pn+1) + "xeq", xeq}, {"p" + std::to_string(pn+1) + "yeq", yeq}});
    else as->manager.set({{"p" + std::to_string(pn+1) + "xeq", xeq}, {"p" + std::to_string(pn+1) + "yeq", yeq}});
}





/*
###  ###  #### ##### #   # ##### ###   ###
#  # #  # #      #   ##  #   #   #  # #   #
###  ###  ###    #   # # #   #   ###  #   #
#    #  # #      #   #  ##   #   #  # #   #
#    #  # #### ##### #   #   #   #  #  ###
*/

void preintro(CompositeScene& cs) {
    shared_ptr<BeaverTNF3DScene> tnfs = make_shared<BeaverTNF3DScene>();
    shared_ptr<TuringMachineScene> tms;
    TuringMachine tm;
    char bb4[28] = "1RB1LB_1LA0LC_1RZ1LD_1RD0RA";
    char tc[14] = "0RB1RA_1LA1RB";
    char bouncer[14] = "0RB1LA_1LA1RB";
    char counter[14] = "0RB0LA_1LA1RB";
    char bigfoot[30] = "1RB2RA1LC_2LC1RB2RB_1R-2LA1LA";

    cs.add_scene(tnfs, "tnfs");
    parse_tmstring(bb4, 4, 2, tm);
    Path p = {7, {0, 1, 2, 3, 6, 9, 11}};
    vec3 center = target_tm(tm, p, 0, 0);

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
        {"zoom", "5"},
        {"camera_distance", "e <zoom> -1 * ^"},
        {"scale_x", "e <zoom> 0.8 * ^"},
        {"scale_y", "e <zoom> 0.5 * ^"},
        {"scale_z", "1"},
        {"brightness_offset", "<zoom> -0.2 *"},
        {"color_source_depth", "<zoom> 1.5 * 3.6 +"}
    });

    stage_macroblock(FileBlock("You are looking at the space of all computer programs."), 1);
    tnfs->manager.transition(MICRO, "zoom", "0");
    cs.render_microblock();


    parse_tmstring(tc, 2, 2, tm);
    p.action[2] = 3;
    p.action[3] = 2;
    p.pathlen = 3;
    vec3 center_2x2s = target_tm(tm, p, 0, 0);
    std::string center_2x2s_x = std::to_string(center_2x2s.x);
    std::string center_2x2s_y = std::to_string(center_2x2s.y);
    std::string center_2x2s_z = std::to_string(center_2x2s.z);
    p.pathlen = 4;
    center = target_tm(tm, p, 0, 0);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 4.5 - 60 / -"},
        {"iterations", "0"}
    });

    stage_macroblock(FileBlock("Most of them just loop forever,"), 3);
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
        {"zoom", "2.7"},
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
        {"zoom", "2"},
        {"highlight_intensity", "0"}
    });
    cs.render_microblock();

    cs.remove_subscene("tms");


    parse_tmstring(bouncer, 2, 2, tm);
    center = target_tm(tm, p, 0, 0);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    cs.manager.set("tms.x", "0.65");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 4.5 - 200 / -"},
        {"iterations", "4.5"}
    });
    stage_macroblock(FileBlock("or bounce from side to side."), 2);

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
        {"zoom", "2.7"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();

    tms->manager.transition(MICRO, "iterations", "4.5");
    tnfs->manager.transition(MICRO, {
        {"target_x", center_2x2s_x},
        {"target_y", center_2x2s_y},
        {"target_z", center_2x2s_z},
        {"zoom", "2"},
        {"highlight_intensity", "0"}
    });
    cs.render_microblock();

    cs.remove_subscene("tms");


    parse_tmstring(counter, 2, 2, tm);
    center = target_tm(tm, p, 0, 0);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    cs.manager.set("tms.x", "0.65");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 4.5 - 400 / -"},
        {"iterations", "4.5"}
    });
    stage_macroblock(FileBlock("Some count to infinity, usually in binary."), 2);

    tms->manager.transition(MICRO, "iterations", "100");
    tnfs->manager.set({
        {"highlight_x", std::to_string(center.x)},
        {"highlight_y", std::to_string(center.y)},
        {"highlight_z", std::to_string(center.z)},
    });
    tnfs->manager.transition(MICRO, {
        {"target_x", std::to_string(center.x)},
        {"target_y", std::to_string(center.y)},
        {"target_z", std::to_string(center.z)},
        {"zoom", "2.7"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();

    tms->manager.transition(MICRO, "iterations", "4.5");
    tnfs->manager.transition(MICRO, {
        {"target_x", center_2x2s_x},
        {"target_y", center_2x2s_y},
        {"target_z", center_2x2s_z},
        {"zoom", "2"},
        {"highlight_intensity", "0"}
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("But occasionally,"), 1);
    cs.manager.transition(MICRO, "tms.x", "0.75");
    tms->manager.transition(MICRO, {
        {"iterations", "0"},
        {"zoom", "-1.5 <iterations> -400 / -"}
    });
    tnfs->manager.set({
        {"target_x", center_2x2s_x + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_y", center_2x2s_y + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"},
        {"target_z", center_2x2s_z + " 0.5 - e 1 1 {microblock_fraction} 4.5 ^ / - ^ 1 - 2 ^ * 0.5 +"}
    });
    tnfs->manager.transition(MICRO, {
        {"spin_offset", "8 {t} 1.5 * -"},
	{"q1", "0"},
	{"qj", "-1"},
        {"zoom", "0"},
        //{"highlight_intensity", "0"},
        {"scale_x", "1"},
        {"scale_y", "1"}
    });
    cs.render_microblock();

    cs.remove_subscene("tms");


    Path pbf = {8, {0, 1, 6, 7, 8, 5, 3, 2}};
    parse_tmstring(bigfoot, 3, 3, tm);
    center = target_tm(tm, pbf, 0, 0);
    tms = make_shared<TuringMachineScene>(tm);
    cs.add_scene(tms, "tms");
    cs.manager.set("tms.x", "0.75");
    tms->manager.set({
        {"ticks_opacity", "0"},
        {"center_y", "<iterations> e <zoom> -1 * ^ e * -"},
        {"zoom", "-1.5 <iterations> 500 / -"},
        {"iterations", "0"}
    });
    stage_macroblock(FileBlock("you might stumble upon Cryptids, the lovecraftian machines gatekeeping the borders of computation and knowability itself."), 1);

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
        {"scale_x", "e 5 4 <zoom> - 2 ^ 0.3125 * - ^"},
        {"scale_y", "e 5 4 <zoom> - 2 ^ 0.3125 * - ^"}
    });
    tnfs->manager.transition(MICRO, {
        {"zoom", "4"},
        {"highlight_intensity", "1"}
    });
    cs.render_microblock();

    tnfs->manager.set({
        {"target_x", std::to_string(center.x)},
        {"target_y", std::to_string(center.y)},
        {"target_z", std::to_string(center.z)}
    });

    stage_macroblock(SilenceBlock(2), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

/*
##### #   # ##### ###   ###
  #   ##  #   #   #  # #   #
  #   # # #   #   ###  #   #
  #   #  ##   #   #  # #   #
##### #   #   #   #  #  ###
*/

void intro(CompositeScene& cs) {
    shared_ptr<BeaverIndividualScene> tms;
    shared_ptr<BeaverIndividualScene> tms2;
    TuringMachine tm;
    char bella[14] = "1RB1LA_0LA1R-";
    char bob[14] = "1RB1LB_0LA1R-";

    std::string latex_string = "Beavers";
    shared_ptr<LatexScene> ls = make_shared<LatexScene>(latex_string, 0.6);

    stage_macroblock(FileBlock("But what even is a computer program?"), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("First, we need to learn how Beavers work."), 3);
    cs.add_scene_fade_in(MICRO, ls, "ls");
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "ls", 0);
    cs.render_microblock();

    parse_tmstring(bella, 2, 2, tm);
    tms = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    tms->default_everything(true, false, false, true, true);
    tms->manager.set({
        {"table_cell_margin", "0.5"},
        {"zoom", "-1"},
        {"spacetime_focus_y", "<iterations> 4 min"},
    });

    stage_macroblock(FileBlock("Bella the Beaver lives along an infinite river, divided into sections."), 2);
    cs.add_scene_fade_in(MICRO, tms, "tms");
    tms->manager.transition(MICRO, {
        {"dir_icon_scale", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("She can build and destroy a dam in each section."), 2);
    tms->manager.transition(MICRO, {
        {"beav_time", "2.6"},
    });
    cs.render_microblock();
    tms->manager.transition(MICRO, {
        {"beav_time", "0"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Here's how Bella settled in when she first found this river."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("She built a dam, walked around for a while, built another dam, and then she retired."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "13"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("To understand what she's doing, let's take a look inside her head."), 3);
    cs.render_microblock();
    cs.render_microblock();
    tms->manager.transition(MICRO, {
        {"beav_time", "0"},
        {"table_col_w", "0.1"},
        {"table_row_h", "0.1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("On any given day, Bella wakes up feeling either Ambitious or Bitter."), 1);
    tms->manager.set({
        {"table_cell_margin", "0"},
    });
    tms->manager.transition(MICRO, {
	{"table_w0", "0.5625"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Depending on her mood, and whether there's a dam in the section she's in, she decides what she will do today."), 1);
    tms->manager.transition(MICRO, {
        {"table_h0", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("On the day she arrived, Bella felt Ambitious."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "1"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("If Bella feels Ambitious and there's no dam, she builds a dam, moves right, and becomes Bitter."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "3.5"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("If Bella feels Bitter and there's no dam, she doesn't build a dam, moves left, and becomes Ambitious."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "6"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("If Bella's Ambitious and there's a dam, she keeps the dam there, moves left, and stays Ambitious."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "8.5"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("Now Bella is Ambitious and there's no dam again, so she builds one, goes right, and becomes Bitter."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "11"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("And finally, if Bella feels Bitter and there's a dam, she decides she has built enough, and retires."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "13"},
    }, false);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "0"},
    });
    cs.render_microblock();

    tms->manager.set({
        //{"beav_time", "5.75"},
        {"iterations", "<beav_time> 2.5 / floor <beav_time> <beav_time> 2.5 / floor 2.5 * - 1.5 - 0 max +"},

        {"dir_icon_scale", "1"},
        {"current_tape_opacity", "1"},
        {"sleep", "1 <iterations> ceil <iterations> floor - - <beav_time> <beav_time> 2.5 / floor 2.5 * - 0.75 < 2 * 1 - *"},

        {"table_col_w", "0.1"},
        {"table_row_h", "0.1"},
	{"table_w0", "0.5625"},
	{"table_h0", "1"},
        {"table_border", "0.06"},
        {"table_line_glow", "0.1"},
        //{"show_all_transitions", "1"},

        {"zoom", "-1"},
        {"center_y", "<iterations> <vertical_step> * 0.5 +"},
    });

    stage_macroblock(FileBlock("Meet Bella's friend, Bob. Bob is very similar to Bella, except for one little change."), 1);
    tms->manager.transition(MICRO, {
        {"beav_time", "5.75"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("When he feels Ambitious and there's a dam, he keeps the dam there and moves left, but he becomes Bitter."), 1);
    parse_tmstring(bob, 2, 2, tm);
    tms->set_tm(tm);
    tms->manager.set("spacetime_focus_y", "<iterations>");
    tms->manager.transition(MICRO, {
        {"beav_time", "8.5"},
    }, false);
    cs.render_microblock();

    stage_macroblock(FileBlock("Hypothetically, if Bob was Bitter in a section of the river with a dam, he would also retire. But unfortunately, that never happens."), 1);
    double global_t = get_global_state("t");
    tms->manager.transition(MICRO, {
        {"beav_time", "{t} " + std::to_string(global_t-0.00001) + " - 1.5 ^ 8.5 +"},
	{"zoom", "-2.5"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Bob is a perfectionist. He will never be content with his work."), 1);
    cs.render_microblock();


    shared_ptr<BeaverGridSpacetimeScene> gs = make_shared<BeaverGridSpacetimeScene>();
    char tc[14] = "1RB1RA_0LA1R-";
    char halt3[14] = "1R-1RB_1LA1R-";
    char equiv_ex1[14] = "1LB0LB_1RB0LB";
    char equiv_ex2[14] = "1LB1R-_1RB0LB";

    stage_macroblock(FileBlock("Can we tell whether a given beaver will retire? Let's try a few more, and let's also keep track of how the river looked in the past."), 4);
    double global_t_2 = get_global_state("t");
    tms->manager.transition(MICRO, "beav_time", std::to_string(pow(global_t_2 - global_t, 1.5)) + " 13 +");
    cs.render_microblock();
    cs.render_microblock();
    tms->manager.set({
        {"opacity_min", "{microblock_fraction} 3 ^ 0.4 *"},
        {"opacity_dropoff", "1.01"},
    });
    tms->manager.transition(MICRO, {
        {"state_icon_scale", "0.5"},
        {"vertical_step", "1"},
        {"opacity_dropoff", "1.4"},
        {"center_y", "<iterations> <vertical_step> * 2.5 -"},
    });
    cs.render_microblock();
    tms->manager.set("opacity_min", "0.4");
    tms->manager.transition(MICRO, {
        {"beav_time", "0"},
        {"center_y", "<iterations> <vertical_step> * 0.5 -"},
    });
    cs.render_microblock();

    parse_tmstring(tc, 2, 2, tm);
    tms2 = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    tms2->default_everything(true, true, true, false, true);
    tms2->manager.set({
        {"state_icon_scale", "0.5"},
        {"vertical_step", "1"},
        {"opacity_min", "0.4"},
        {"opacity_dropoff", "1.4"},

        {"dir_icon_scale", "1"},
        {"current_tape_opacity", "1"},
        //{"sleep", "1 <iterations> ceil <iterations> floor - - <beav_time> <beav_time> 2.5 / floor 2.5 * - 0.75 < 2 * 1 - *"},

        {"table_col_w", "0.1"},
        {"table_row_h", "0.1"},
	{"table_w0", "0.5625"},
	{"table_h0", "1"},
        {"table_border", "0.06"},
        {"table_line_glow", "0.1"},
        //{"show_all_transitions", "1"},

        {"zoom", "-2"},
        {"center_y", "<iterations> <vertical_step> * 0.5 -"},
    });
    cs.add_scene(tms2, "tms2");
    tms->manager.set("iterations", "0");

    stage_macroblock(FileBlock("This beaver is a perfectionist too. This one retired immediately."), 2);
    cs.manager.set({
        {"tms.x", "0.5 {macroblock_fraction} 2 * floor + {macroblock_fraction} 2 * {macroblock_fraction} 2 * floor - 5 * 1 min -"},
        {"tms2.x", "1.5 {macroblock_fraction} 2 * floor - {macroblock_fraction} 2 * {macroblock_fraction} 2 * floor - 5 * 1 min -"},
    });
    tms2->manager.set("iterations", "50 {macroblock_fraction} 0.1 - 0 max 1.5 ^ *");
    cs.render_microblock();
    parse_tmstring(halt3, 2, 2, tm);
    tms->set_tm(tm);
    tms->manager.set({
        {"center_y", "0.5"},
        {"iterations", "5 {microblock_fraction} 0.2 - 0 max *"},
        {"zoom", "-1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("It would be faster to look at them all side by side."), 1);
    cs.manager.set({
        {"tms.x", "0.5"},
        {"tms2.x", "0.5 e <zoom> ^ -"},
        {"zoom", "1 1 <zoom_param> / -"},
        {"zoom_param", "1"},
    });
    tms->manager.set({
        {"iterations", "4"},
        {"w", "e [zoom] ^"},
        {"h", "e [zoom] ^"},
    });
    tms2->manager.set({
        {"iterations", "40"},
        {"w", "e [zoom] ^"},
        {"h", "e [zoom] ^"},
    });
    cs.manager.transition(MICRO, "zoom_param", "0");
    cs.render_microblock();
    cs.manager.set({
        {"tms.opacity", "0"},
        {"tms2.opacity", "0"},
        {"zoom", "0"},
    });

    stage_macroblock(FileBlock("With two emotions, a beaver has 4 instructions, and each instruction has 9 options for what it can be. That means there are 6561 beavers who are always either Ambitious or Bitter."), 7);
    cs.render_microblock();
    latex_string = "4";
    ls->jump_latex(latex_string);
    cs.fade_subscene(MICRO, "ls", 1);
    cs.render_microblock();
    cs.render_microblock();
    latex_string = "9^4";
    ls->begin_latex_transition(MICRO, latex_string);
    cs.render_microblock();
    latex_string = "9^4 = 6561";
    ls->begin_latex_transition(MICRO, latex_string);
    cs.render_microblock();
    latex_string = "6561";
    ls->begin_latex_transition(MICRO, latex_string);
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Let's put their rivers on a grid. I will highlight the beavers who have retired."), 3);
    // TODO: bunch of features for grid scene
    cs.fade_subscene(MICRO, "ls", 0);
    cs.render_microblock();
    gs->manager.set("iterations", "0");
    cs.add_scene_fade_in(MICRO, gs, "gs");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("After one day, one ninth of the beavers have already retired."), 1);
    gs->manager.set("iterations", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("Day two goes by, even more are eliminated."), 1);
    gs->manager.set("iterations", "2");
    cs.render_microblock();

    stage_macroblock(FileBlock("Day three, day four, day five... Do you think all the remaining ones are perfectionists, or is someone secretly working on a masterpiece?"), 1);
    gs->manager.set("log_iters", "0.8");
    gs->manager.set("iterations", "e <log_iters> ^");
    gs->manager.transition(MICRO, "log_iters", "2.5");
    cs.render_microblock();

    stage_macroblock(FileBlock("Okay, I'll spoil it. All remaining beavers here are perfectionists."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Looking at this grid, you might notice some patterns, large groups of beavers behaving identically."), 3);
    gs->manager.transition(MICRO, {
        {"zoom", "0.8"},
        {"center_x", "5 6 /"},
        {"center_y", "5 6 /"},
    });
    cs.render_microblock();
    gs->manager.transition(MICRO, {
        {"zoom", "1.7"},
        {"center_x", "17 18 /"},
        {"center_y", "11 18 /"},
    });
    cs.render_microblock();
    gs->manager.transition(MICRO, {
        {"zoom", "2.6"},
        {"center_x", "15 18 /"},
        {"center_y", "11 18 /"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("This is because, starting on an empty river, some of their instructions remain unused. If two beavers differ only in instructions they never actually use, then their behavior is exactly the same."), 4);
    cs.move_to_back("gs");
    cs.fade_subscene(MICRO, "gs", 0.15);
    cs.fade_subscene(MICRO, "tms", 1);
    parse_tmstring(equiv_ex1, 2, 2, tm);
    tms->set_tm(tm);
    tms->manager.set({
        {"iterations", "0"},
        {"zoom", "-2"},
        {"center_y", "<iterations> <vertical_step> * 0.5 -"},
        {"w", "3 32 /"},
        {"h", "1 6 /"},
    });
    tms->manager.transition(MICRO, {
        {"w", "1"},
        {"h", "1"},
    });
    cs.render_microblock();
    tms->manager.transition(MICRO, {
        {"iterations", "5.999"},
    });
    cs.render_microblock();
    cs.fade_subscene(MICRO, "tms2", 1);
    parse_tmstring(equiv_ex2, 2, 2, tm);
    tms2->set_tm(tm);
    tms->manager.set({
        {"show_all_transitions", "2"},
    });
    tms2->manager.set({
        {"iterations", "5.999"},
        {"zoom", "-2"},
        {"w", "0.001"},
        {"center_y", "<iterations> <vertical_step> * 0.5 -"},
        {"show_all_transitions", "2"},
    });
    tms->manager.transition(MICRO, {
        {"w", "0.5"},
    });
    tms2->manager.transition(MICRO, {
        {"w", "0.5"},
    });
    cs.manager.set("tms2.x", "1.02");
    cs.manager.transition(MICRO, {
        {"tms.x", "0.24"},
        {"tms2.x", "0.76"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Researchers consider them to actually be the same beaver, and don't fill in the unused instructions."), 2);
    tms->manager.transition(MICRO, {
        {"w", "1"},
    });
    tms2->manager.transition(MICRO, {
        {"w", "1"},
    });
    cs.manager.transition(MICRO, {
        {"tms.x", "0.5"},
        {"tms2.x", "0.5"},
    });
    cs.render_microblock();
    cs.manager.set("tms2.opacity", "0");
    tms->manager.set("show_all_transitions", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("If we account for this, there are far fewer beavers with two emotions - only 297."), 1);
    // TODO: remove 3 things from gs in this case
    gs->manager.set("zoom", "0");
    gs->manager.transition(MICRO, {
        {"grid_w", "20"},
        {"grid_h", "15"},
        {"center_x", "0.5"},
        {"center_y", "0.5"},
    });
    cs.manager.set("gs.opacity", "1");
    cs.manager.set("tms.opacity", "0");
    cs.manager.set("tms2.opacity", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("If we also consider beavers that are mirror images of each other to be the same, we can cut that down to 149."), 1);
    gs->manager.transition(MICRO, {
        {"grid_w", "15"},
        {"grid_h", "10"},
    });
    // remove 4 things from gs in this case
    cs.render_microblock();

    stage_macroblock(FileBlock("And since we only want to know who retires and who doesn't, the beavers with no unused or retiring instructions are all solved too, leaving only 61."), 1);
    gs->manager.transition(MICRO, {
        {"grid_w", "9"},
        {"grid_h", "7"},
    });
    // remove 2 things from gs in this case
    cs.render_microblock();

    stage_macroblock(FileBlock("19 of them have already retired, and the remaining 42 are easy enough to go through by hand and see that they are perfectionists, all moving periodically."), 1);
    gs->manager.transition(MICRO, {
        {"grid_w", "7"},
        {"grid_h", "6"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Beavers with 2 emotions and 1 type of dam are a nice example to illustrate what the researchers are trying to do, and how they do it."), 1);
    cs.fade_subscene(MICRO, "gs", 0);
    // TODO: use affinescene instead
    cs.fade_subscene(MICRO, "ls", 1);
    latex_string = "\\text{Solving beavers}";
    ls->jump_latex(latex_string);
    cs.render_microblock();

    stage_macroblock(FileBlock("We were able to reduce the search space from 6561 beavers to only 61, and then all we needed was some automated way to tell when a beaver is a perfectionist, and a little work by hand."), 3);
    latex_string = "\\text{Solving beavers\n1. Reduce by equivalences (6561\to 61)}";
    //ls->begin_latex_transition(MICRO, latex_string);
    cs.render_microblock();
    latex_string = "\\text{Solving beavers\n1. Reduce by equivalences (6561\to 61)\n2. Automation (61\to 42)}";
    //ls->begin_latex_transition(MICRO, latex_string);
    cs.render_microblock();
    latex_string = "\\text{Solving beavers\n1. Reduce by equivalences (6561\to 61)\n2. Automation (61\to 42)\n3. Solved by hand (42\to 0)}";
    //ls->begin_latex_transition(MICRO, latex_string);
    cs.render_microblock();

    cs.remove_all_subscenes();
}

/*
#  # #  ### #  # #### ###    ###   ##  #   #  ##  # #  #  ###
#  # # #    #  # #    #  #   #  # #  # ## ## #  # # ## # #
#### # # ## #### ###  ###    #  # #  # # # # #### # # ##  ##
#  # # #  # #  # #    #  #   #  # #  # #   # #  # # #  #    #
#  # #  ### #  # #### #  #   ###   ##  #   # #  # # #  # ###
*/

void higher_domains(CompositeScene& cs) {
    TuringMachine tm;
    shared_ptr<BeaverIndividualScene> tms = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    shared_ptr<BeaverIndividualScene> tms2 = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    shared_ptr<AffineScene> as;
    char bouncer[21] = "1RB1RA_0LC1R-_0RA1LC";
    char counter[21] = "1LB1R-_1RC1LB_0LA0RC";
    char fib[28] = "1RB0LC_0LA0RB_1RA1RD_1LA1R-";
    char bouncy_counter[28] = "1RB0LA_0RC0RA_0LD1RA_1LA1R-";
    char fractal_tm[28] = "1RB1R-_0RC0RB_1LC0LD_1RA0LD";
    char shift_overflow[35] = "1LC1RD_1RE1R-_0LD0LC_1RB0RA_1RA1LE";
    char skelet1[35] = "1RB1RD_1LC0RC_1RA1LD_0RE0LB_1R-1RC";
    char skelet17[35] = "1RB1R-_0LC1RE_0LD1LC_1RA1LB_0RB0RA";
    int pn;
    int tn;

    cs.add_scene(tms, "tms");
    cs.add_scene(tms2, "tms2");
    tms->default_everything(true, true, true, false, true);
    tms2->default_everything(true, true, true, false, true);

    // leave this (mostly) out?
    /*
    But some beavers have more emotions, or know how to build different types of dams, which affect their decisions differently.
    When exploring them, we'd like to avoid duplicates, so we will structure our visualization around only filling in instructions that the beaver actually uses.
    Then we should start from the beaver who doesn't have any instructions filled in, represented by this block.
    This beaver immediately tries to use an empty instruction, so we need to fill it.
    A retiring instruction wouldn't really give us anything new, just a beaver who retires immediately, so let's just consider the other possible instructions.
    It might seem like there are infinitely many - the beaver could build any type of dam (or not build a dam), and switch to any emotion.
    But right now, there's no difference between the types of dams, because they haven't been used yet, so let's allow only building a wooden dam, or no dam at all.
    Similarly, all emotions except Ambition are kind of the same, so let's allow only Ambition and Bitterness.
    The third decision to make is the direction of movement, where the only options are left and right.
    */

    /*
    This means we have 3 axes, and we can split our block along those axes to get the new beavers.
    For example, all beavers on the left side start by moving left, and all beavers on the right side start by moving right. Similarly, the four beavers on the front side stay Ambitious.
    They don't try to use any empty instructions, they're already perfectionists, so we get no new beavers by filling in their instructions. Let's make their cuboids empty.
    The other four immediately try to use another empty instruction, so we will need to fill that one and split their cuboids accordingly.
    Now that Bitterness has been used, it's distinct from the other emotions, but we can just choose Curiosity as the next representative of the infinity of equivalent emotions.
    Emptying the perfectionist blocks again, we get this shape. As we continue filling instructions and removing perfectionists, this fractal pops out.
    Zooming in, the blocks become stretched out, but we can squish space to account for that.
    */

    stage_macroblock(FileBlock("With 3 emotions, we see new kinds of perfectionists: bouncers and counters."), 1);
    parse_tmstring(bouncer, 3, 2, tm);
    tms->set_tm(tm);
    parse_tmstring(counter, 3, 2, tm);
    tms2->set_tm(tm);
    cs.manager.set({
        {"tms.x", "0.25"},
        {"tms.zoom", "-3.1"},
        {"tms.stfx", "0"},
        {"tms.stfy", "15"},
        {"tms.vs", "0.7"},
        {"tms2.x", "0.75"},
        {"tms2.zoom", "-3.1"},
        {"tms2.stfx", "0"},
        {"tms2.stfy", "15"},
        {"tms2.vs", "0.7"},
    });
    tms->manager.set({
        {"w", "0.5"},
        {"zoom", "[tms.zoom]"},
        {"center_x", "[tms.stfx]"},
        {"spacetime_focus_y", "[tms.stfy]"},
        {"vertical_step", "[tms.vs]"},
    });
    tms2->manager.set({
        {"w", "0.5"},
        {"zoom", "[tms2.zoom]"},
        {"center_x", "[tms2.stfx]"},
        {"spacetime_focus_y", "[tms2.stfy]"},
        {"vertical_step", "[tms2.vs]"},
    });
    tms->manager.transition(MICRO, {
        {"iterations", "30"},
    });
    tms2->manager.transition(MICRO, {
        {"iterations", "30"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("So far, all counters count in binary, but that changes when a fourth emotion is added into the mix:"), 3);
    cs.manager.transition(MICRO, {
        {"tms.x", "-0.25"},
        {"tms2.x", "0.5"},
    });
    tms2->manager.transition(MICRO, {
        {"w", "1"},
        {"iterations", "100"},
    });
    // show binary overlay (affine scene)
    as = make_shared<AffineScene>(icons, icons_whnm, icons_len);
    cs.add_scene(as, "as");
    as->manager.set({
        {"r", "0"},
        {"g", "0"},
        {"b", "0"},
        {"stcx", "0.5625 e [tms2.zoom] ^ *"},
        {"stcy", "[tms2.vs] e [tms2.zoom] ^ *"},
    });
    int t=3;
    int d;
    int l=1;
    int aaaaaaaaa=1;
    for (int i=1; i<17; i++) {
        d = 3;
        for (int j=0; j<l; j++) {
            d += (d&1)*(1+((i>>j)&1));
            pn = as->num_points;
            tn = as->text_anchors.size();
            as->add_point();
            as->add_text(pn, {latex_color(0xff000000, std::to_string((i>>j)&1))});
            as->manager.set({
                {"p" + std::to_string(pn) + "xeq", "<p" + std::to_string(pn) + "x> [tms2.stfx] - <stcx> * [tms2.x] +"},
                {"p" + std::to_string(pn) + "yeq", "<p" + std::to_string(pn) + "y> [tms2.stfy] - <stcy> * [tms2.y] +"},
                {"p" + std::to_string(pn) + "x", std::to_string(-j-1)},
                {"p" + std::to_string(pn) + "y", std::to_string(t)},
                {"t" + std::to_string(tn) + "h", "0.05"},
                {"t" + std::to_string(tn) + "o", "<opacity" + std::to_string(aaaaaaaaa) + ">"},
            });
        }
        pn = as->num_points;
        tn = as->text_anchors.size();
        as->add_point();
        as->add_text(pn, {latex_color(0xff000000, std::to_string(i))});
        as->manager.set({
            {"p" + std::to_string(pn) + "xeq", "<p" + std::to_string(pn) + "x> [tms2.stfx] - <stcx> * [tms2.x] +"},
            {"p" + std::to_string(pn) + "yeq", "<p" + std::to_string(pn) + "y> [tms2.stfy] - <stcy> * [tms2.y] +"},
            {"p" + std::to_string(pn) + "x", std::to_string(3)},
            {"p" + std::to_string(pn) + "y", std::to_string(t)},
            {"t" + std::to_string(tn) + "h", "0.07"},
            {"t" + std::to_string(tn) + "o", "<opacity" + std::to_string(aaaaaaaaa) + ">"},
        });
        aaaaaaaaa = d&1;
        l += d&1;
        d += d&1;
        t += d;
    }
    as->manager.set({
        {"opacity0", "0"},
        {"opacity1", "0"},
    });
    as->manager.transition(MICRO, {
        {"opacity0", "1"},
        {"opacity1", "1"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms2.vs", "0.4"},
        {"tms2.zoom", "-3.7"},
        {"tms2.stfy", "50"},
    });
    cs.render_microblock();
    // highlight powers of 2
    as->manager.transition(MICRO, {
        {"opacity0", "0.2"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("This beaver is counting in a number system based on the Fibonacci sequence!"), 4);
    parse_tmstring(fib, 4, 2, tm);
    tms->set_tm(tm);
    cs.manager.set({
        {"tms.x", "1.5"},
        //{"tms.vs", "0.7"},
        //{"tms.stfy", "15"},
    });
    tms->manager.set({
        {"w", "1"},
        {"iterations", "0"},
        //{"vertical_step", "0.7"},
        //{"center_y", "<iterations> 13 - <vertical_step> *"},
    });
    cs.manager.transition(MICRO, {
        {"tms.x", "0.5"},
        {"tms2.x", "-0.5"},
    });
    tms->manager.transition(MICRO, {
        {"iterations", "30"},
    });
    cs.render_microblock();
    // show fibbinary overlay (affine scene)
    as->reset();
    as->manager.set({
        {"r", "0"},
        {"g", "0"},
        {"b", "0"},
        {"stcx", "0.5625 e [tms.zoom] ^ *"},
        {"stcy", "[tms.vs] e [tms.zoom] ^ *"},
    });
    t=13;
    l=2;
    int ifib=3;
    aaaaaaaaa=1;
    for (int i=1; i<17; i++) {
        d = 1;
        for (int j=0; j<l; j++) {
            d += (d&1)*(1+(((ifib>>j)&1)||j==0));
            pn = as->num_points;
            tn = as->text_anchors.size();
            as->add_point();
            as->add_text(pn, {latex_color(0xff000000, std::to_string((ifib>>j)&1))});
            as->manager.set({
                {"p" + std::to_string(pn) + "xeq", "<p" + std::to_string(pn) + "x> [tms.stfx] - <stcx> * [tms.x] +"},
                {"p" + std::to_string(pn) + "yeq", "<p" + std::to_string(pn) + "y> [tms.stfy] - <stcy> * [tms.y] +"},
                {"p" + std::to_string(pn) + "x", std::to_string(-j-1)},
                {"p" + std::to_string(pn) + "y", std::to_string(t)},
                {"t" + std::to_string(tn) + "h", "0.05"},
                {"t" + std::to_string(tn) + "o", "<opacity" + std::to_string(aaaaaaaaa) + ">"},
            });
        }
        pn = as->num_points;
        tn = as->text_anchors.size();
        as->add_point();
        as->add_text(pn, {latex_color(0xff000000, std::to_string(i))});
        as->manager.set({
            {"p" + std::to_string(pn) + "xeq", "<p" + std::to_string(pn) + "x> [tms.stfx] - <stcx> * [tms.x] +"},
            {"p" + std::to_string(pn) + "yeq", "<p" + std::to_string(pn) + "y> [tms.stfy] - <stcy> * [tms.y] +"},
            {"p" + std::to_string(pn) + "x", std::to_string(3)},
            {"p" + std::to_string(pn) + "y", std::to_string(t)},
            {"t" + std::to_string(tn) + "h", "0.07"},
            {"t" + std::to_string(tn) + "o", "<opacity" + std::to_string(aaaaaaaaa) + ">"},
        });
        as->manager.transition(MICRO, {
            {"t" + std::to_string(tn) + "o", "<opacity" + std::to_string(aaaaaaaaa) + ">"},
        });
        aaaaaaaaa = d&1;
        l += d&1;
        d += d&1;
        int aaaaa = ifib&1;
        ifib += 2-(ifib&1)+(1<<(d/2-2));
        d = 2*d+2+4*(1-aaaaa);
        t += d;
    }
    as->manager.set({
        {"opacity0", "0"},
        {"opacity1", "0"},
    });
    as->manager.transition(MICRO, {
        {"opacity0", "1"},
        {"opacity1", "1"},
    });
    tms->manager.transition(MICRO, {
        {"iterations", "300"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms.vs", "0.3"},
        {"tms.zoom", "-4.5"},
        {"tms.stfy", "150"},
    });
    cs.render_microblock();
    // highlight fibonacci numbers
    as->manager.transition(MICRO, {
        {"opacity0", "0.2"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("And some beavers even mix the counting with the bouncing, while others build fractals"), 3);
    parse_tmstring(bouncy_counter, 4, 2, tm);
    tms2->set_tm(tm);
    cs.manager.set({
        {"tms2.x", "1.25"},
        {"tms2.zoom", "-3.1"},
        {"tms2.stfx", "0"},
        {"tms2.stfy", "15"},
        {"tms2.vs", "0.7"},
    });
    tms2->manager.set({
        {"w", "0.5"},
        {"iterations", "0"},
        //{"vertical_step", "0.7"},
        //{"center_y", "<iterations> 13 - <vertical_step> *"},
    });
    cs.manager.transition(MICRO, {
        {"tms.x", "-0.5"},
        {"tms2.x", "0.25"},
    });
    cs.render_microblock();
    as->reset();
    parse_tmstring(fractal_tm, 4, 2, tm);
    tms->set_tm(tm);
    cs.manager.set({
        {"tms.x", "1.75"},
        {"tms.zoom", "-3.1"},
        {"tms.stfx", "0"},
        {"tms.stfy", "15"},
        {"tms.vs", "0.7"},
    });
    tms->manager.set({
        {"w", "0.5"},
        {"iterations", "0"},
    });
    cs.manager.transition(MICRO, {
        {"tms.x", "0.75"},
    });
    tms2->manager.transition(MICRO, {
        {"iterations", "30"},
    });
    // show binary overlay?
    cs.render_microblock();
    tms->manager.transition(MICRO, {
        {"iterations", "30"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("With 5 emotions, we also get the first shift-overflow counters, who most often count on two sides, but momentarily do something else when one of the counters overflows, usually with a chance to retire."), 4);
    cs.fade_subscene(MICRO, "tms", 0);
    cs.fade_subscene(MICRO, "tms2", 0);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "tms", 1);
    parse_tmstring(shift_overflow, 5, 2, tm);
    tms->set_tm(tm);
    cs.manager.set({
        {"tms.x", "0.5"},
        {"tms.iterations", "50"},
        {"tms.vs", "0.4"},
        {"tms.stfy", "25"},
    });
    tms->manager.set({
        {"w", "1"},
        {"iterations", "[tms.iterations]"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms.x", "0.5"},
        {"tms.iterations", "250"},
        {"tms.vs", "0.4"},
        {"tms.zoom", "-3.8"},
        {"tms.stfy", "200"},
    });
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(FileBlock("Beaver researchers have designed programs, so-called deciders, that detect if a beaver is a certain kind of perfectionist."), 3);
    cs.fade_subscene(MICRO, "tms", 0);
    cs.fade_subscene(MICRO, "tms2", 0);
    cs.render_microblock();
    // timeline titled "Deciders" using affine scene
    as->reset();
    as->add_point();
    as->add_text(0, {"Deciders"});
    as->manager.set({
        {"r", "1"},
        {"g", "1"},
        {"b", "1"},
        {"p0x", "0.5"},
        {"p0y", "0.15"},
        {"t0o", "0"},
        {"t0h", "0.2"},
    });
    as->manager.transition(MICRO, {
        {"t0o", "1"},
    });
    cs.render_microblock();
    float timeline_half_text_height = 0.035;
    as->add_point();
    as->add_point();
    as->add_point();
    as->add_point();
    as->add_line(3,4);
    as->manager.set({
        {"p1x", "0"},
        {"p1y", "0.5"},
        {"p1o", "0"},
        {"p2x", "1"},
        {"p2y", "0.5"},
        {"p2o", "0"},
        {"p3x", "0.5"},
        {"p3y", "0.5"},
        {"p4x", "0.5"},
        {"p4y", "0.5"},
    });
    as->manager.transition(MICRO, {
        {"p3x", "-0.1"},
        {"p4x", "1.1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("It started with Shen Lin's decider for beavers who move periodically, with which Lin was able to classify all beavers with 3 emotions."), 1);
    float year = 1965;
    as->add_point({1,2});
    as->add_text(as->num_points-1, {"\\text{3 emotions solved}"});
    as->add_text(as->num_points-1, {std::to_string((int)(year))});
    as->add_text(as->num_points-1, {"\\text{Periodic beavers}"});
    as->add_text(as->num_points-1, {"\\text{by Shen Lin}"});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "w1", std::to_string(2034-year)},
        {"p" + std::to_string(as->num_points-1) + "w2", std::to_string(year-1954)},
        {"t" + std::to_string(as->text_anchors.size()-4) + "y", std::to_string(-3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-4) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-4) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "y", std::to_string(-timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "y", std::to_string(timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "y", std::to_string(3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "h", std::to_string(2*timeline_half_text_height)},
    });
    as->manager.transition(MICRO, {
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-4) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Then Allen Brady made deciders for bouncers and counters, and this was enough to get through the remaining 4-emotion beavers by hand."), 1);
    // ~550000 TNF, 5820 post-(Lin deciders (TC & backwards)), 218 post-(Brady deciders (bouncers & counters)), 0 post-hand
    year = 1983;
    as->add_point({1,2});
    as->add_text(as->num_points-1, {"\\text{4 emotions solved}"});
    as->add_text(as->num_points-1, {std::to_string((int)(year))});
    as->add_text(as->num_points-1, {"\\text{Bouncers \\& Counters}"});
    as->add_text(as->num_points-1, {"\\text{by Allen Brady}"});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "w1", std::to_string(2034-year)},
        {"p" + std::to_string(as->num_points-1) + "w2", std::to_string(year-1954)},
        {"t" + std::to_string(as->text_anchors.size()-4) + "y", std::to_string(-3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-4) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-4) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "y", std::to_string(-timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "y", std::to_string(timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "y", std::to_string(3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "h", std::to_string(2*timeline_half_text_height)},
    });
    as->manager.transition(MICRO, {
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-4) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "1"},
    });
    cs.render_microblock();

    // TODO: some ad hoc visualization for this probably
    // actually a good example to show would be "to the right of the beaver, there's a contiguous block of dams followed by only water" or something
    stage_macroblock(FileBlock("But deciders don't have to detect what the beaver actually does."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Sometimes they only prove that the beaver always preserves some pattern, but any hypothetical situation where the beaver is retired could be traced back to a time when that pattern wasn't present, which is impossible."), 1);
    cs.render_microblock();

    // back to timeline
    stage_macroblock(FileBlock("After adding a few such deciders to their toolkit, researchers solved most beavers with 5 emotions, bringing their number down from 16 *trillion* to less than a hundred."), 1);
    year = 2003;
    as->add_point({1,2});
    as->add_text(as->num_points-1, {std::to_string((int)(year))});
    as->add_text(as->num_points-1, {"\\text{Closed position set}"});
    as->add_text(as->num_points-1, {"\\text{by Georgi Georgiev}"});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "w1", std::to_string(2034-year)},
        {"p" + std::to_string(as->num_points-1) + "w2", std::to_string(year-1954)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "y", std::to_string(-timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "y", std::to_string(timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "y", std::to_string(3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "h", std::to_string(2*timeline_half_text_height)},
    });
    as->manager.transition(MICRO, {
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "1"},
    });
    year = 2022;
    as->add_point({1,2});
    as->add_text(as->num_points-1, {std::to_string((int)(year))});
    as->add_text(as->num_points-1, {"\\text{CTL \\& FAR}"});
    as->add_text(as->num_points-1, {"\\text{by Shawn Ligocki \\& Justin Blanchard}"});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "w1", std::to_string(2034-year)},
        {"p" + std::to_string(as->num_points-1) + "w2", std::to_string(year-1954)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "y", std::to_string(-timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "y", std::to_string(timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "y", std::to_string(3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "h", std::to_string(2*timeline_half_text_height)},
    });
    as->manager.transition(MICRO, {
        {"p1x", "-0.35"},
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-3) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Then they reduced it by hand to 2, and after months of work, 0."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("In July 2024, four decades after researchers solved the 4-emotion beavers, they finally did the same for the 5-emotion ones."), 1);
    year = 2024.05;
    as->add_point({1,2});
    as->add_text(as->num_points-1, {"\\text{5 emotions solved}"});
    as->add_text(as->num_points-1, {std::to_string((int)(year))});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "w1", std::to_string(2034-year)},
        {"p" + std::to_string(as->num_points-1) + "w2", std::to_string(year-1954)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "y", std::to_string(-3*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "h", std::to_string(2*timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "y", std::to_string(-timeline_half_text_height)},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "0"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "h", std::to_string(2*timeline_half_text_height)},
    });
    as->manager.transition(MICRO, {
        {"p1x", "-3"},
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-2) + "o", "1"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("And just a month earlier, they realized that would be the end of their journey."), 1);
    year = 2023.95;
    as->add_point({1,2});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "w1", std::to_string(2034-year)},
        {"p" + std::to_string(as->num_points-1) + "w2", std::to_string(year-1954)},
        {"p" + std::to_string(as->num_points-1) + "o", "0"},
    });
    as->manager.transition(MICRO, {
        {"p1x", "-83.5"},
        {"p2x", "12.5"},
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("With up to 5 emotions, all beavers slipping through the cracks in the deciders were fully predictable."), 2);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    cs.manager.set("as.opacity", "1");
    as->reset();
    cs.render_microblock();

    // skelet 1 (leave out?)
    /*stage_macroblock(FileBlock("Even this beaver fortunately entered a periodic behavior when, using some clever tricks, it was simulated to quindecillions of steps."), 1);
    cs.render_microblock();*/

    stage_macroblock(FileBlock("But after Ambition, Bitterness, Curiosity, Disgust and Excitement came the sixth emotion: Fear."), 2);
    as->add_point();
    for (int i=0; i<5; i++) {
        as->add_icon(0);
        as->manager.set({
            {"i" + std::to_string(as->icon_anchors.size()-1) + "x", "<i" + std::to_string(as->icon_anchors.size()-1) + "mag> <i" + std::to_string(as->icon_anchors.size()-1) + "angle> cos * 0.5625 *"},
            {"i" + std::to_string(as->icon_anchors.size()-1) + "y", "<i" + std::to_string(as->icon_anchors.size()-1) + "mag> <i" + std::to_string(as->icon_anchors.size()-1) + "angle> sin *"},
            {"i" + std::to_string(as->icon_anchors.size()-1) + "h", "<i" + std::to_string(as->icon_anchors.size()-1) + "mag> 0.8 *"},
            {"i" + std::to_string(as->icon_anchors.size()-1) + "mag", "<sp> " + std::to_string(i) + " 3 / - 0 max 0.3 min e <zoom> ^ *"},
            {"i" + std::to_string(as->icon_anchors.size()-1) + "angle", "<sp> " + std::to_string(i) + " pi 2.5 / * +"},
            {"i" + std::to_string(as->icon_anchors.size()-1) + "id", std::to_string(icons_whnm.w + i)},
        });
    }
    as->manager.set({
        {"p0x", "0.5"},
        {"p0y", "0.5"},
        {"sp", "{t} " + std::to_string(get_global_state("t")) + " - 2 *"},
        {"zoom", "0"},
    });
    cs.render_microblock();
    as->add_icon(0);
    as->manager.set({
        {"i" + std::to_string(as->icon_anchors.size()-1) + "h", "0"},
        {"i" + std::to_string(as->icon_anchors.size()-1) + "id", std::to_string(icons_whnm.w + 5)},
    });
    as->manager.transition(MICRO, {
        {"i" + std::to_string(as->icon_anchors.size()-1) + "h", "0.3 e <zoom> ^ *"},
        {"zoom", "{t} " + std::to_string(get_global_state("t")) + " - 2 +"},
    });
    cs.render_microblock();

    // TODO: TNF fractal zoom on antihydra
    stage_macroblock(FileBlock("And it was brought to the researchers by beavers known as Cryptids."), 2);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_all_subscenes();
}

/*
 ### ###  #   # ###  ##### # ###   ###
#    #  #  # #  #  #   #   # #  # #
#    ###    #   ###    #   # #  #  ##
#    #  #   #   #      #   # #  #    #
 ### #  #   #   #      #   # ###  ###
*/

void cryptids(CompositeScene& cs) {
    TuringMachine tm;
    shared_ptr<BeaverIndividualScene> tms0 = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    shared_ptr<BeaverIndividualScene> tms1 = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    shared_ptr<BeaverIndividualScene> tms2 = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    shared_ptr<BeaverIndividualScene> tms3 = make_shared<BeaverIndividualScene>(tm, icons, icons_whnm, icons_len);
    vector<shared_ptr<BeaverIndividualScene>> tms = {tms0, tms1, tms2, tms3};
    shared_ptr<AffineScene> as = make_shared<AffineScene>(icons, icons_whnm, icons_len);
    shared_ptr<LatexScene> ls;
    vector<string> labels = {"\\text{Antihydra}", "\\text{???}", "\\text{Space Needle}", "\\text{BMO #1}", "\\text{Lucy's Moonlight}", "\\text{Bigfoot}", "\\text{Champion?}", "\\text{Wily Coyote}", "\\ ", "\\text{Hydra}"};
    vector<string> tmstrs = {"1RB1RA_0LC1LE_1LD1LC_1LA0LB_1LF1RE_1R-0RA", "1RB1RA_0LC0RF_1LD1LC_1RE1LB_1R-0RA_1RF0RD", "1RB1LA_1LC0RE_1LF1LD_0RB0LA_1RC1RE_1R-0LD", "1RB1RE_1LC0RA_0RD1LB_1R-1RC_1LF1RE_0LB0LE", "1RB0RD_0RC1RE_1RD0LA_1LE1LC_1RF0LD_1R-0RA", "1RB2RA1LC_2LC1RB2RB_1R-2LA1LA", "1RB2LC1RC_2LC1R-2RB_2LA0LB0RA", "1RB2LA1LA_2LA0RA2RC_1R-0LC2RA", "1RB1LB2LC_1LA2RB1RB_1R-0LA2LA", "1RB3RB1R-3LA1RA_2LA3RA4LB0LB0LA"};
    vector<float> spect_pos = {0.1, 0.9, 0.1+11.0/80, 0.9-11.0/80, 0.1, 0.1, 0.5, 0.9, 0.9, 0.1};
    int antihydra = 0; int chaos_with_buffer = 1; int space_needle = 2; int bmo1 = 3; int lucy = 4; int bigfoot = 5; int probv_champ = 6; int wily_coyote = 7; int coyote_like = 8; int hydra = 9;
    vector<int> shown_tms;
    int spect_start;
    int pn; int ln; int tn; int in;
    int antihydra_table_n = 9;
    float antihydra_table_row_h = 0.1;
    int antihydra_t;
    int antihydra_d;
    double global_t;

    for (int i=0; i<4; i++) {
        cs.add_scene(tms[i], "tms" + std::to_string(i));
        cs.manager.set({
            {"tms" + std::to_string(i) + ".opacity", "0"},
            {"tms" + std::to_string(i) + ".w", "1"},
            {"tms" + std::to_string(i) + ".h", "1"},
            {"tms" + std::to_string(i) + ".iterations", "0"},
            {"tms" + std::to_string(i) + ".vs", "0.2"},
            {"tms" + std::to_string(i) + ".zoom", "-4"},
            {"tms" + std::to_string(i) + ".stfx", "0"},
            {"tms" + std::to_string(i) + ".stfy", "130"},
        });
        tms[i]->default_everything(false, true);
        tms[i]->manager.set({
            {"w", "[tms" + std::to_string(i) + ".w]"},
            {"h", "[tms" + std::to_string(i) + ".h]"},
            {"iterations", "[tms" + std::to_string(i) + ".iterations]"},
            {"vertical_step", "[tms" + std::to_string(i) + ".vs]"},
            {"zoom", "[tms" + std::to_string(i) + ".zoom]"},
            {"center_x", "[tms" + std::to_string(i) + ".stfx]"},
            {"spacetime_focus_y", "[tms" + std::to_string(i) + ".stfy]"},
        });
    }
    cs.add_scene(as, "as");
    cs.manager.set("as.opacity", "0");


    stage_macroblock(FileBlock("This is Antihydra. Its behavior is quite predictable on a low level."), 2);
    parse_tmstring(tmstrs[antihydra], 6, 2, tm);
    tms0->set_tm(tm);
    cs.manager.set({
        {"tms0.iterations", "{macroblock_fraction} 2 ^ 500 *"},
        {"as.opacity", "0"},
    });
    cs.fade_subscene(MICRO, "tms0", 1);
    tms0->manager.set({
        {"show_all_transitions", "2"},
    });
    tms0->manager.transition(MICRO, {
        {"table_col_w", "0.05"},
        {"table_row_h", "0.05"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.set({
        {"tms0.iterations", "1800"},
    });

    stage_macroblock(FileBlock("On the right side, there's a bouncy part, which expands until it hits this left edge, at which point the bouncing starts over from further away."), 3);
    // zoom in
    cs.manager.transition(MICRO, {
        {"tms0.zoom", "-2.7"},
        {"tms0.stfx", "10"},
        {"tms0.stfy", "200"},
    });
    cs.render_microblock();
    // zoom out
    cs.manager.transition(MICRO, {
        {"tms0.zoom", "-3"},
        {"tms0.stfx", "-4"},
        {"tms0.stfy", "380"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms0.zoom", "-3.3"},
        {"tms0.stfx", "0"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Right after each reset of the bouncy part, the entire situation is characterized by two numbers: The thickness of the edge, and the distance between it and the bouncy part."), 4);
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_point();
    as->add_point({0,1});
    as->add_point({0,2});
    as->add_point({1,2});
    as->add_text(3,{latex_color(0xff000000, "t")});
    as->add_text(4,{latex_color(0xff000000, "d")});
    as->add_line(0,1);
    as->reorder_layers({LINE, POINT, TEXT});
    as->manager.set({
        {"r", "0"},
        {"g", "0"},
        {"b", "0"},
        {"stcx", "0.5625 e [tms0.zoom] ^ *"},
        {"stcy", "[tms0.vs] e [tms0.zoom] ^ *"},
        {"p0xeq", "<p0x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p0yeq", "<p0y> [tms0.stfy] - <stcy> * 0.5 +"},
        {"p1xeq", "<p1x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p1yeq", "<p1y> [tms0.stfy] - <stcy> * 0.5 +"},
        {"p0x", "-14"},
        {"p0y", "417"},
        {"p0t", "0.01"},
        {"p1x", "17"},
        {"p1y", "417"},
        {"p1t", "0.01"},
        {"p2w0", "24 7 /"},
        {"p2t", "0.01"},
        {"p3y", "0.05"},
        //{"p3o", "0.3"},
        {"p3r", "1"},
        {"p3g", "1"},
        {"p3b", "1"},
        {"p3t", "0.03"},
        {"p4y", "0.05"},
        //{"p4o", "0.3"},
        {"p4r", "1"},
        {"p4g", "1"},
        {"p4b", "1"},
        {"p4t", "0.03"},
        {"l0t", "0.005"},
        {"t0o", "0"},
        {"t1o", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"t0o", "1"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"t1o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Now we just need to know the rules that determine what these two numbers will turn into when the next reset happens."), 4);
    as->add_point();
    as->add_text(5,{latex_color(0xff000000, "(t,d)\\ \\to\\ (?,?)")});
    as->manager.set({
        {"p5x", "0.5"},
        {"p5y", "0.5"},
        {"t2o", "0"},
        {"t2h", "0.4"},
    });
    as->manager.transition(MICRO, {
        {"t2o", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("In each bounce, the bouncy part expands left by 2 river sections, and it expands right by 1 river section."), 3);
    cs.manager.transition(MICRO, {
        {"tms0.vs", "0.7"},
        {"tms0.zoom", "-3"},
        {"tms0.stfx", "15"},
        {"tms0.stfy", "442.25"},
    });
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_point();
    as->add_point({2,3});
    as->add_point({3,4});
    as->add_point({5,6});
    as->add_point({6,7});
    as->add_text(8, {latex_color(0xff000000, "+2")});
    as->add_text(9, {latex_color(0xff000000, "+2")});
    as->add_text(10, {latex_color(0xff000000, "+1")});
    as->add_text(11, {latex_color(0xff000000, "+1")});
    as->manager.set({
        {"p0x", "18 1 15 / - <p0y> 420 - sqrt 1.04 * -"},
        {"p0yeq", "2 3 /"},
        {"p0y", "<p0yeq> 0.5 - <stcy> / [tms0.stfy] +"},
        {"p1x", "17 1 15 / - <p1y> 420 - sqrt 0.52 * +"},
        {"p1yeq", "2 3 /"},
        {"p1y", "<p1yeq> 0.5 - <stcy> / [tms0.stfy] +"},
        {"p2x", "12.6"},
        {"p2xeq", "<p2x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p2yeq", "2 3 /"},
        {"p2t", "0.01"},
        {"p2o", "<p0x> <p2x> <"},
        {"p3x", "10.6"},
        {"p3xeq", "<p3x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p3yeq", "2 3 /"},
        {"p3t", "0.01"},
        {"p3o", "<p0x> <p3x> <"},
        {"p3r", "<r>"},
        {"p3g", "<g>"},
        {"p3b", "<b>"},
        {"p4x", "8.6"},
        {"p4xeq", "<p4x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p4yeq", "2 3 /"},
        {"p4o", "0"},
        {"p5x", "19.6"},
        {"p5xeq", "<p5x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p5yeq", "2 3 /"},
        {"p5t", "0.01"},
        {"p5o", "<p1x> <p5x> >"},
        {"p6x", "20.6"},
        {"p6xeq", "<p6x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p6yeq", "2 3 /"},
        {"p6t", "0.01"},
        {"p6o", "<p1x> <p6x> >"},
        {"p7x", "21.6"},
        {"p7xeq", "<p7x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p7yeq", "2 3 /"},
        {"p7o", "0"},
        {"p8y", "0.03"},
        {"p8r", "1"},
        {"p8g", "1"},
        {"p8b", "1"},
        {"p9y", "0.03"},
        {"p9r", "1"},
        {"p9g", "1"},
        {"p9b", "1"},
        {"p10y", "0.03"},
        {"p10r", "1"},
        {"p10g", "1"},
        {"p10b", "1"},
        {"p11y", "0.03"},
        {"p11r", "1"},
        {"p11g", "1"},
        {"p11b", "1"},
        {"t0o", "0"},
        {"t1o", "0"},
        {"t2o", "0"},
        {"t3o", "0"},
        {"t4o", "0"},
        {"t5o", "0"},
        {"t6o", "0"},
    });
    cs.render_microblock();
    as->manager.set({
        //{"p8o", "<p2x> <p0x> - 0.15 * 0.3 min"},
        //{"p9o", "<p3x> <p0x> - 0.15 * 0.3 min"},
        //{"p10o", "<p1x> <p5x> - 0.3 * 0.3 min"},
        //{"p11o", "<p1x> <p6x> - 0.3 * 0.3 min"},
        {"t3o", "<p2x> <p0x> - 2 /"},
        {"t3h", "0.06"},
        {"t4o", "<p3x> <p0x> - 2 /"},
        {"t4h", "0.06"},
        {"t5o", "<p1x> <p5x> -"},
        {"t5h", "0.06"},
        {"t6o", "<p1x> <p6x> -"},
        {"t6h", "0.06"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.stfy", "501 e <tms0.zoom> -1 * ^ <tms0.vs> / 6 / -"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("So at all times, the location from which the bouncy part started expanding splits the bouncy part in a 2:1 ratio."), 2);
    // zoom out to include start of the bouncer, lower a vertical line from it
    as->add_point();
    as->add_point();
    as->add_line(12,13);
    as->manager.set({
        {"p12xeq", "<p12x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p12yeq", "<p12y> [tms0.stfy] - <stcy> * 0.5 +"},
        {"p13xeq", "<p13x> [tms0.stfx] - <stcx> * 0.5 +"},
        {"p13yeq", "<p13y> [tms0.stfy] - <stcy> * 0.5 +"},
        {"p12x", "17.5"},
        {"p12y", "420"},
        {"p12t", "0.01"},
        {"p13x", "17.5"},
        {"p13y", "420"},
        {"p13o", "0"},
        {"l1t", "0.005"},
    });
    as->manager.transition(MICRO, {
        {"p2o", "0"},
        {"p3o", "0"},
        {"p5o", "0"},
        {"p6o", "0"},
        {"p8o", "0"},
        {"p9o", "0"},
        {"p10o", "0"},
        {"p11o", "0"},
        {"t3o", "0"},
        {"t4o", "0"},
        {"t5o", "0"},
        {"t6o", "0"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.vs", "0.13"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"p13y", "<p0y>"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("By the time of the next reset, it will have expanded right by about half the starting distance to the edge."), 2);
    // when it hits the edge, write latex "d" and "(1/2)d" in the appropriate places
    as->add_point({0,1});
    as->add_point({0,1});
    as->add_point({0,1});
    as->add_text(14, {latex_color(0xff000000,"d")});
    as->add_text(15, {latex_color(0xff000000,"\\frac{1}{2}d")});
    as->add_text(16, {latex_color(0xff000000,"\\frac{3}{2}d")});
    as->manager.set({
        {"p14w0", "2"},
        {"p15w1", "5"},
        {"p14y", "0.05"},
        {"p15y", "0.05"},
        {"p16y", "0.05"},
        {"p16r", "1"},
        {"p16g", "1"},
        {"p16b", "1"},
        {"t7o", "0"},
        {"t8o", "0"},
        {"t9o", "0"},
    });
    as->manager.transition(MICRO, {
        {"t7o", "1"},
        {"t8o", "1"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.vs", "0.1"},
        {"tms0.zoom", "-4.7"},
        //{"tms0.stfy", "440 e <tms0.zoom> -1 * ^ <tms0.vs> / 4 / +"},
        {"tms0.stfy", "729"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Then the bouncy part resets, starting from the right end, which means the distance between the edge and the bouncy part gets roughly multiplied by 3/2."), 3);
    // combine d and (1/2)d into (3/2)d
    as->manager.set({
        {"coll", "{microblock_fraction} 3 ^"},
        {"p14w0", "2 <coll> 2 / -"},
        {"p14w1", "1 <coll> 2 / +"},
        {"p15w0", "1 <coll> 2 * +"},
        {"p15w1", "5 <coll> 2 * -"},
    });
    as->manager.transition(MICRO, {
        {"p12o", "0"},
        {"l1o", "0"},
    });
    cs.render_microblock();
    as->draw_order.push_back({POINT, 16});
    as->manager.set({
        {"coll", "1"},
        {"coll2", "1 {microblock_fraction} - 2 ^ 5 * 4 -"},
        {"p16o", "1 <coll2> <coll2> 1.25 * <coll2> 0 < * - -"},
        {"p16t", "1 <coll2> <coll2> 1.25 * <coll2> 0 < * - - 0.06 *"},
        {"p17o", "<coll2> 0 < <coll2> 4 / 1 + * 2 ^"},
        {"p17t", "<coll2> -12 / 0 max"},
        {"t7o", "<coll2> <coll2> 0 > *"},
        {"t8o", "<coll2> <coll2> 0 > *"},
        {"t9o", "1 <coll2> <coll2> 0 > * -"},
    });
    cs.render_microblock();
    as->manager.set({
        {"p16o", "0"},
        {"t7o", "0"},
        {"t8o", "0"},
        {"t9o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(0.5), 1);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();


    stage_macroblock(FileBlock("To see what's happening with the thickness of the edge, let's zoom in on the moment of impact."), 2);
    // zoom in
    cs.manager.transition(MICRO, {
        {"tms0.vs", "0.7"},
    });
    begin_zoom(cs, "tms0.zoom", "tms0.stfx", "tms0.stfy", -3, -5, 880);
    cs.render_microblock();
    cs.manager.set({
        {"tms0.zoom", "-3"},
        {"tms0.stfx", "-5"},
        {"tms0.stfy", "880"},
    });
    // fade in affinescene with length indicator of edge thickness
    cs.fade_subscene(MICRO, "as", 1);
    add_point_spacetime(as, "tms0");
    add_point_spacetime(as, "tms0");
    as->add_line(0,1);
    add_point_spacetime(as, "tms0");
    add_point_spacetime(as, "tms0");
    as->add_line(2,3);
    add_point_spacetime(as, "tms0");
    as->add_line(3,4);
    as->add_point({3,4});
    as->add_text(5, {"\\textcolor{#000000}{-1}"});
    as->reorder_layers({LINE, POINT, TEXT});
    as->manager.set({
        {"p0x", "-13.45"},
        {"p0y", "875"},
        {"p1x", "-7.45"},
        {"p1y", "875"},
        {"p2x", "-13.45"},
        {"p2y", "875"},
        {"p3x", "-7.45"},
        {"p3y", "875"},
        {"p4x", "-7.45"},
        {"p4y", "875"},
        {"p4o", "0"},
        {"p4r", "1"},
        {"l2o", "0"},
        {"l2r1", "1"},
        {"l2r2", "1"},
        {"t0y", "0.03"},
        {"t0o", "0"},
        {"t0h", "0.06"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("In this situation, Antihydra just shrinks the edge by 1 river section and resets the bouncy part."), 2);
    // make copy of length indicator move downwards and shrink by 1
    as->manager.transition(MICRO, {
        {"p2y", "885"},
        {"p3x", "-8.45"},
        {"p3y", "885"},
        {"p4y", "885"},
        {"p4o", "1"},
        {"l2o", "1"},
        {"t0o", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("But last time, the beaver instead expanded the edge by 2 river sections."), 3);
    // split screen, make copies of length indicator for the other scene as well
    tms1->set_tm(tm);
    cs.manager.set({
        {"tms1.opacity", "1"},
        {"tms1.x", "1.25"},
        {"tms1.w", "0.5"},
        {"tms1.iterations", "600"},
        {"tms1.vs", "0.7"},
        {"tms1.zoom", "-3.2"},
        {"tms1.stfx", "-5"},
        {"tms1.stfy", "385"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.x", "0.25"},
        {"tms0.w", "0.5"},
        {"tms1.x", "0.75"},
    });
    tms0->manager.transition(MICRO, {
        {"table_col_w", "0"},
        {"table_row_h", "0"},
    });
    cs.render_microblock();
    add_point_spacetime(as, "tms1");
    add_point_spacetime(as, "tms1");
    as->add_line(6,7);
    add_point_spacetime(as, "tms1");
    add_point_spacetime(as, "tms1");
    as->add_line(8,9);
    add_point_spacetime(as, "tms1");
    as->add_point({8,10});
    as->add_text(11, {"\\textcolor{#000000}{+2}"});
    as->manager.set({
        {"p6x", "-10.45"},
        {"p6y", "375"},
        {"p7x", "-6.45"},
        {"p7y", "375"},
        {"p8x", "-10.45"},
        {"p8y", "375"},
        {"p9x", "-6.45"},
        {"p9y", "375"},
        {"p10x", "-10.45"},
        {"p10y", "375"},
        {"t1y", "0.03"},
        {"t1o", "0"},
        {"t1h", "0.06"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"p8x", "-13.45"},
        {"p8y", "395"},
        {"p9x", "-7.45"},
        {"p9y", "395"},
        {"p10x", "-11.45"},
        {"p10y", "395"},
        {"t1o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Whether Antihydra shrinks the edge by 1 or expands it by 2 depends on how many river sections remain between the edge and the bouncy part after all the bounces happen -"), 3);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();
    // highlight the space between the bouncy part and the edge
    cs.manager.transition(MICRO, {
        {"tms0.zoom", "-2.5"},
        {"tms0.stfx", "-5.4"},
        {"tms0.stfy", "872"},
        {"tms1.zoom", "-2.5"},
        {"tms1.stfy", "369"},
    });
    cs.fade_subscene(MICRO, "as", 1);
    add_point_spacetime(as, "tms0");
    add_point_spacetime(as, "tms0");
    as->add_line(0,1);
    add_point_spacetime(as, "tms1");
    add_point_spacetime(as, "tms1");
    as->add_line(2,3);
    as->manager.set({
        {"p0x", "-6.45"},
        {"p0y", "871"},
        {"p1x", "-4.45"},
        {"p1y", "871"},
        {"p2x", "-5.45"},
        {"p2y", "368"},
        {"p3x", "-4.45"},
        {"p3y", "368"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("how much of the distance remains when 2 has been subtracted as many times as possible while staying positive."), 2);
    // column of "-2"s dropping on "d-2*[number]" where the number is increasing
    cs.manager.transition(MICRO, {
        {"tms0.x", "0.5"},
        {"tms0.w", "1"},
        {"tms0.zoom", "-1.5"},
        {"tms1.x", "1.25"},
    });
    as->add_point();
    as->add_text(4, {"\\textcolor{#000000}{d-2\\cdot", "}"});
    as->manager.set({
        {"p4x", "0.5"},
        {"p4y", "0.5"},
        {"t0o", "0"},
        {"t0h", "0.2"},
        {"t0v0", "<t0v> floor 0 max"},
        {"t0v", "-4"},
    });
    for (int i=0; i<5; i++) {
        as->add_text(4, {"\\textcolor{#000000}{-2}"});
        as->manager.set({
            {"t" + std::to_string(i+1) + "x", "0.05"},
            {"t" + std::to_string(i+1) + "y", "<t0v> <t0v> floor 0 max - " + std::to_string(i+1) + " - 0.125 *"},
            {"t" + std::to_string(i+1) + "o", "0"},
            {"t" + std::to_string(i+1) + "h", "0.2"},
        });
    }
    as->manager.transition(MICRO, {
        {"t0o", "1"},
    });
    cs.render_microblock();
    for (int i=0; i<5; i++) as->manager.transition(MICRO, "t" + std::to_string(i+1) + "o", "10");
    as->manager.set({
        {"t0v", "e {microblock_fraction} 5 * ^ 5 -"},
    });
    cs.render_microblock();
    for (int i=0; i<6; i++) {
        as->draw_order.pop_back();
        as->text_latex.pop_back();
    }

    stage_macroblock(FileBlock("It depends on the remainder after dividing the distance by 2! Or more simply, the parity of the distance."), 4);
    // number turns into floor((d-1)/2), then "d-2*floor((d-1)/2)" latex transitions into "2-(d modulo 2)"
    ls = make_shared<LatexScene>("\\textcolor{#000000}{d-2\\cdot\\lfloor\\frac{d-1}{2}\\rfloor}", 0.4);
    cs.add_scene(ls, "ls");
    cs.render_microblock();
    // TODO: figure out how to make it remain black during the transition
    ls->begin_latex_transition(MICRO, "\\textcolor{#000000}{2-(d\\text{ modulo }2)}");
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();


    stage_macroblock(FileBlock("So, when researchers filled in the details, these are the rules they found."), 1);
    cs.fade_subscene(MICRO, "tms0", 0);
    cs.fade_subscene(MICRO, "tms1", 0);
    cs.fade_subscene(MICRO, "as", 0);
    cs.fade_subscene(MICRO, "ls", 0);
    cs.render_microblock();
    as->reset();
    as->manager.set({
        {"r", "1"},
        {"g", "1"},
        {"b", "1"},
    });

    //stage_macroblock(FileBlock("If the distance is even, multiply it by 3/2, and increase the edge thickness by 2."), 3);
    stage_macroblock(FileBlock("If the distance is even, increase the edge thickness by 2."), 2);
    // latex even case fades in
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_point();
    as->add_text(1, {"d\\text{ is even}:"});
    as->add_point();
    as->add_text(2, {"(t,d)\\ \\to\\ (t+2,\\lfloor\\frac{3}{2}d\\rfloor)"});
    as->reorder_layers({TEXT, POINT});
    as->manager.set({
        {"p0x", "0.879"},
        {"p0t", "0.07"},
        {"p0r", "0"},
        {"p0g", "0"},
        {"p0b", "68 256 /"},
        {"p0m11", "0"},
        {"p0s", "1"},
        {"p1x", "0.25"},
        {"p1y", "1 3 /"},
        {"p2x", "0.75"},
        {"p2y", "1 3 /"},
        {"t0h", "0.2"},
        {"t1h", "0.2"},
    });
    cs.render_microblock();
    cs.render_microblock();

    //stage_macroblock(FileBlock("If the distance is odd, multiply it by 3/2, round down, and decrease the edge thickness by 1."), 3);
    stage_macroblock(FileBlock("If the distance is odd, decrease the edge thickness by 1."), 2);
    // latex odd case fades in
    as->add_point();
    as->add_text(3, {"d\\text{ is odd}:"});
    as->add_point();
    as->add_text(4, {"(t,d)\\ \\to\\ (t-1,\\lfloor\\frac{3}{2}d\\rfloor)"});
    as->reorder_layers({TEXT, POINT});
    as->manager.set({
        {"p3x", "0.25"},
        {"p3y", "2 3 /"},
        {"p4x", "0.75"},
        {"p4y", "2 3 /"},
        {"t2o", "0"},
        {"t2h", "0.2"},
        {"t3o", "0"},
        {"t3h", "0.2"},
    });
    as->manager.transition(MICRO, {
        {"t2o", "1"},
        {"t3o", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("In both cases, multiply the distance by 3/2 and round down."), 2);
    as->manager.transition(MICRO, {
        {"p0o", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("And if the edge thickness goes below 0, Antihydra retires."), 3);
    // latex retirement case fades in
    as->add_point();
    as->add_text(5, {"t<0:"});
    as->add_point();
    as->add_icon(6);
    as->manager.set({
        {"p5x", "0.25"},
        {"p5y", "1"},
        {"p6x", "0.75"},
        {"p6y", "1"},
        {"t4o", "0"},
        {"t4h", "0.2"},
        {"i0o", "0"},
        {"i0h", "0.2"},
        {"i0id", std::to_string(icons_len-1)},
    });
    as->manager.transition(MICRO, {
        {"p1y", "0.25"},
        {"p2y", "0.25"},
        {"p3y", "0.5"},
        {"p4y", "0.5"},
        {"p5y", "0.75"},
        {"p6y", "0.75"},
        {"t4o", "1"},
        {"i0o", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Starting from distance 8 and edge thickness 0, will Antihydra retire?"), 3);
    // trajectory table shows up
    as->manager.transition(MICRO, {
        {"p1x", "0.2"},
        {"p2x", "0.55"},
        {"p3x", "0.2"},
        {"p4x", "0.55"},
        {"p5x", "0.2"},
        {"p6x", "0.55"},
    });
    cs.render_microblock();
    pn = as->num_points;
    ln = as->line_endpoints.size();
    tn = as->text_anchors.size();
    as->add_point();
    as->add_point({pn});
    as->add_point({pn,pn+1});
    as->add_point({pn,pn+2});
    as->add_point({pn+1,pn+2});
    as->add_point({pn});
    as->add_point({pn+1});
    as->add_point({pn+2});
    as->add_point({pn,pn+2});
    as->add_point({pn+1,pn+2});
    as->add_line(pn,pn+5);
    as->add_line(pn+1,pn+6);
    as->add_line(pn+2,pn+7);
    as->add_line(pn+8,pn+9);
    antihydra_t = 0;
    antihydra_d = 8;
    for (int i=0; i<antihydra_table_n; i++) {
        as->add_text(pn+3, {i<antihydra_table_n-1 ? std::to_string(antihydra_t) : "..."});
        as->add_text(pn+4, {i<antihydra_table_n-1 ? std::to_string(antihydra_d) : "..."});
        as->add_line(pn,pn+1);
        as->manager.set({
            {"t" + std::to_string(tn+2*i  ) + "y", std::to_string((i+0.5)*antihydra_table_row_h)},
            {"t" + std::to_string(tn+2*i  ) + "o", "<tp> " + std::to_string(i) + " -"},
            {"t" + std::to_string(tn+2*i+1) + "y", std::to_string((i+0.5)*antihydra_table_row_h)},
            {"t" + std::to_string(tn+2*i+1) + "o", "<tp> " + std::to_string(i) + " -"},

            {"l" + std::to_string(ln+i+4) + "y", std::to_string(i*antihydra_table_row_h)},
            {"l" + std::to_string(ln+i+4) + "o", "<tp> " + std::to_string(i) + " >"},
        });
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.set({
        {"tp", "-1"},
        {"p" + std::to_string(pn  ) + "x", "1 0.228125 -"},
        {"p" + std::to_string(pn  ) + "y", "0.05"},
        {"p" + std::to_string(pn+1) + "x", "0.2"},
        {"p" + std::to_string(pn+5) + "y", "<tp> " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+6) + "y", "<tp> " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+7) + "y", "<tp> " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+8) + "w" + std::to_string(pn  ), "<tp> <tp> 1 - ceil -"},
        {"p" + std::to_string(pn+8) + "w" + std::to_string(pn+2), "<tp> ceil <tp> -"},
        {"p" + std::to_string(pn+9) + "w" + std::to_string(pn+1), "<tp> <tp> 1 - ceil -"},
        {"p" + std::to_string(pn+9) + "w" + std::to_string(pn+2), "<tp> ceil <tp> -"},
        {"l" + std::to_string(ln  ) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+1) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+2) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+3) + "y", "<tp> ceil " + std::to_string(antihydra_table_row_h) + " *"},
    });
    for (int i=0; i<10; i++) as->manager.set("p" + std::to_string(pn+i) + "o", "0");
    as->manager.transition(MICRO, {
        {"tp", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This is a clean math problem, you don't need to understand beavers in order to try to solve it. In fact, similar problems have been researched a lot before."), 3);
    // latex trajectory
    as->manager.transition(MICRO, {
        {"tp", std::to_string(antihydra_table_n)},
    });
    cs.render_microblock();
    // fade out rules
    as->manager.transition(MICRO, {
        {"t0o", "0"},
        {"t1o", "0"},
        {"t2o", "0"},
        {"t3o", "0"},
        {"t4o", "0"},
        {"i0o", "0"},
    });
    cs.render_microblock();
    as->add_point();
    as->add_point({pn+10});
    as->add_point({pn+10});
    as->add_line(pn+10, pn+11);
    as->add_line(pn+10, pn+12);
    as->manager.set({
	{"p" + std::to_string(pn+10) + "x", "0.1"},
	{"p" + std::to_string(pn+10) + "y", "0.8"},
	{"p" + std::to_string(pn+10) + "o", "0"},
	{"p" + std::to_string(pn+11) + "x", "0.5"},
	{"p" + std::to_string(pn+11) + "o", "0"},
	{"p" + std::to_string(pn+12) + "y", "-0.6"},
	{"p" + std::to_string(pn+12) + "o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "s", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "s", "1"},
    });
    as->manager.transition(MICRO, {
	{"l" + std::to_string(as->line_endpoints.size()-2) + "o", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("If we plot the edge thickness over time, it looks random,"), 1);
    // graph t
    antihydra_t = 0;
    antihydra_d = 8;
    float so = 0.2;
    for (int i=0; i<30; i++) {
        as->add_point();
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "xeq", "<p" + std::to_string(pn+3) + "xeq> <p" + std::to_string(as->num_points-1) + "x> <gp> " + std::to_string(so*i) + " - 0 max 1 min smoothlerp"},
            {"p" + std::to_string(as->num_points-1) + "x", "<p" + std::to_string(pn+10) + "xeq> <p" + std::to_string(pn+11) + "xeq> <p" + std::to_string(pn+10) + "xeq> - " + std::to_string(0.034*i) + " * +"},
            {"p" + std::to_string(as->num_points-1) + "yeq", "<p" + std::to_string(pn+3) + "yeq> <t" + std::to_string(tn+2*min(i,antihydra_table_n-1)) + "y> + <p" + std::to_string(as->num_points-1) + "y> <gp> " + std::to_string(so*i) + " - 0 max 1 min smoothlerp"},
            {"p" + std::to_string(as->num_points-1) + "y", "<p" + std::to_string(pn+10) + "yeq> <p" + std::to_string(pn+12) + "yeq> <p" + std::to_string(pn+10) + "yeq> - " + std::to_string(0.04*antihydra_t) + " * +"},
            {"p" + std::to_string(as->num_points-1) + "o", "<gp> " + std::to_string(so*i-0.5) + " - 0 max 1 min"},
        });
        if (i>0) {
            as->add_line(as->num_points-2, as->num_points-1);
            as->manager.set({
                {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "<gp> " + std::to_string(so*i-0.5) + " - 0 max 1 min"},
            });
        }
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.set("gp", "-0.5");
    as->manager.transition(MICRO, "gp", "6.8");
    cs.render_microblock();

    stage_macroblock(FileBlock("and there is one particularly well-known math problem which deals with the unpredictability of how the parity of a number changes if you repeatedly multiply it by 3 and divide by 2:"), 1);
    // TODO: zoom out to show a scaled up collatz tree
    // alternatively, since there won't be time for such fancy stuff, just put in some png
    cs.render_microblock();

    stage_macroblock(FileBlock("The Collatz conjecture, about which Paul Erdős famously said 'Mathematics may not be ready for such problems.'"), 1);
    // TODO: latex "collatz conjecture" fades in, then paul erdos with the quote fades in
    cs.render_microblock();

    stage_macroblock(FileBlock("With six emotions, we are granted sufficiently expressive behavior so as to express problems yet unsolved by modern mathematics. That's what cryptids are."), 1);
    // TODO: fade out, then show antihydra art from kayo
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();


    stage_macroblock(FileBlock("But wait, we can at least make a probabilistic analysis!"), 1);
    // TODO: antihydra art fades out
    cs.render_microblock();
    as->reset();

    stage_macroblock(FileBlock("While Antihydra's actual behavior is fully deterministic, as far as we can tell, there are no patterns in the thickness that don't also appear in a random walk."), 2);
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_point({0});
    as->add_point({0});
    as->add_point({0});
    as->add_line(0, 2);
    as->add_line(0, 3);
    as->manager.set({
        {"p0x", "0.1"},
        {"p0y", "0.8"},
        {"p0o", "0"},
        {"p1x", "0.5"},
        {"p1y", "-0.6"},
        {"p1o", "0"},
        {"p2x", "0.5"},
        {"p2o", "0"},
        {"p3y", "-0.6"},
        {"p3o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "s", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "s", "1"},
        {"gp", "0"},
    });
    antihydra_t = 0;
    antihydra_d = 8;
    int antihydra_graph_n = 30;
    for (int i=0; i<antihydra_graph_n; i++) {
        as->add_point();
        if (i>0) as->add_line(as->num_points-2, as->num_points-1);
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "xeq", "<p0xeq> <p1x> " + std::to_string(antihydra_graph_n) + " / <p" + std::to_string(as->num_points-1) + "x> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "xeq> " + std::to_string(i) + " <gp> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "yeq", "<p0yeq> <p1y> " + std::to_string(antihydra_graph_n) + " / <p" + std::to_string(as->num_points-1) + "y> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "yeq> " + std::to_string(i) + " <gp> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "x", std::to_string(i)},
            {"p" + std::to_string(as->num_points-1) + "y", std::to_string(antihydra_t)},
            //{"p" + std::to_string(as->num_points-1) + "o", "0"},
        });
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.transition(MICRO, {
        {"gp", std::to_string(antihydra_graph_n-1)},
    });
    cs.render_microblock();
    pn = as->num_points;
    ln = as->line_endpoints.size();
    tn = as->text_anchors.size();
    in = as->icon_anchors.size();
    as->add_point();
    as->add_point({pn});
    as->add_point({pn,pn+1});
    as->add_point({pn,pn+2});
    as->add_point({pn+1,pn+2});
    as->add_point({pn});
    as->add_point({pn+1});
    as->add_point({pn+2});
    as->add_point({pn,pn+2});
    as->add_point({pn+1,pn+2});
    as->add_line(pn,pn+5);
    as->add_line(pn+1,pn+6);
    as->add_line(pn+2,pn+7);
    as->add_line(pn+8,pn+9);
    antihydra_t = 0;
    antihydra_d = 8;
    global_t = get_global_state("t");
    for (int i=0; i<antihydra_table_n; i++) {
        as->add_text(pn+3, {i<antihydra_table_n-1 ? std::to_string(antihydra_t) : "..."});
        if (i<antihydra_table_n-1) {
            as->add_point({pn+4});
            add_coin(as, as->num_points-1);
        }
        else as->add_text(pn+4, {"..."});
        as->add_line(pn,pn+1);
        as->manager.set({
            {"t" + std::to_string(tn+3*i) + "y", std::to_string((i+0.5)*antihydra_table_row_h)},
            {"t" + std::to_string(tn+3*i) + "o", "<tp> " + std::to_string(i) + " -"},
            {(i<antihydra_table_n-1 ? ("p" + std::to_string(pn+10+i)) : ("t" + std::to_string(tn+3*i+1))) + "y", std::to_string((i+0.5)*antihydra_table_row_h)},
            {(i<antihydra_table_n-1 ? ("c" + std::to_string(pn+10+i)) : ("t" + std::to_string(tn+3*i+1))) + "o", "<tp> " + std::to_string(antihydra_table_n+0.1f*i) + " -"},
            {"c" + std::to_string(pn+10+i) + "r", "0.045"},
            {"c" + std::to_string(pn+10+i) + "a", "{t} " + std::to_string(global_t+11+0.2f*i) + " - 0 min " + std::to_string(15+0.2f*i) + " * " + std::to_string(antihydra_d & 1) + " pi * +"},

            {"l" + std::to_string(ln+i+4) + "y", std::to_string(i*antihydra_table_row_h)},
            {"l" + std::to_string(ln+i+4) + "o", "<tp> " + std::to_string(i) + " >"},
        });
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.set({
        {"tp", "-1"},
        {"p" + std::to_string(pn  ) + "x", "1 0.228125 -"},
        {"p" + std::to_string(pn  ) + "y", "0.05"},
        {"p" + std::to_string(pn+1) + "x", "0.2"},
        {"p" + std::to_string(pn+5) + "y", "<tp> " + std::to_string(antihydra_table_n) + " min " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+6) + "y", "<tp> " + std::to_string(antihydra_table_n) + " min " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+7) + "y", "<tp> " + std::to_string(antihydra_table_n) + " min " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+8) + "w" + std::to_string(pn  ), "<tp> " + std::to_string(antihydra_table_n) + " min <tp> " + std::to_string(antihydra_table_n) + " min 1 - ceil -"},
        {"p" + std::to_string(pn+8) + "w" + std::to_string(pn+2), "<tp> " + std::to_string(antihydra_table_n) + " min ceil <tp> " + std::to_string(antihydra_table_n) + " min -"},
        {"p" + std::to_string(pn+9) + "w" + std::to_string(pn+1), "<tp> " + std::to_string(antihydra_table_n) + " min <tp> " + std::to_string(antihydra_table_n) + " min 1 - ceil -"},
        {"p" + std::to_string(pn+9) + "w" + std::to_string(pn+2), "<tp> " + std::to_string(antihydra_table_n) + " min ceil <tp> " + std::to_string(antihydra_table_n) + " min -"},
        {"l" + std::to_string(ln  ) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+1) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+2) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+3) + "y", "<tp> ceil " + std::to_string(antihydra_table_n) + " min " + std::to_string(antihydra_table_row_h) + " *"},
    });
    for (int i=0; i<10; i++) as->manager.set("p" + std::to_string(pn+i) + "o", "0");
    as->manager.transition(MICRO, {
        {"tp", std::to_string(antihydra_table_n)},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("We can pretend that each change in the edge thickness is determined by a simple coin flip, rather than the parity of the distance."), 1);
    // bring up the table again, but replace the d column with coins flipping
    as->manager.transition(MICRO, {
        {"tp", std::to_string(antihydra_table_n+2)},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Then we can calculate the probability that, from some initial value, the edge thickness will eventually go below 0."), 3);
    // graph reshapes to drop to -1, then graph expands to the whole screen while table goes offscreen and latex t_0 appears, then latex "P(t_0 -> -1) = 1/phi^{t_0+1}" appears below it
    as->manager.transition(MICRO, {
        {"p1x", "0.8"},
        {"p1y", "-1.5"},
        {"p2x", "0.8"},
        {"p" + std::to_string(pn) + "x", "1.5"},
    });
    antihydra_t = 4;
    antihydra_d = 49;
    for (int i=0; i<antihydra_graph_n; i++) {
        as->manager.transition(MICRO, {
            {"p" + std::to_string(i+4) + "y", std::to_string(antihydra_t)},
        });
        antihydra_t += 2 - ((antihydra_d&3)<3) * 3;
        antihydra_d += antihydra_d >> 2;
    }
    cs.render_microblock();
    for (int i=pn; i<as->num_points; i++) as->manager.set("p" + std::to_string(i) + "o", "0");
    for (int i=ln; i<as->line_endpoints.size(); i++) as->manager.set("l" + std::to_string(i) + "o", "0");
    for (int i=tn; i<as->text_anchors.size(); i++) as->manager.set("t" + std::to_string(i) + "o", "0");
    tn = as->text_anchors.size();
    as->add_text(4, {"t_0"});
    as->manager.set({
        {"t" + std::to_string(tn) + "x", "-0.02"},
        {"t" + std::to_string(tn) + "o", "0"},
    });
    as->manager.transition(MICRO, {
        //{"p0x", "0.25"},
        {"p0y", "0.7"},
        {"t" + std::to_string(tn) + "o", "1"},
    });
    cs.render_microblock();
    cs.remove_subscene("ls");
    ls = make_shared<LatexScene>("P(t_0 \\to -1) = \\frac{1}{\\varphi^{t_0+1}}", 0.4);
    cs.add_scene_fade_in(MICRO, ls, "ls");
    cs.manager.set("ls.y", "0.85");
    as->manager.transition(MICRO, {
        {"p" + std::to_string(antihydra_graph_n+3) + "g", std::to_string(antihydra_graph_n-1) + " <gp> - 0 max 1 min"},
        {"p" + std::to_string(antihydra_graph_n+3) + "b", std::to_string(antihydra_graph_n-1) + " <gp> - 0 max 1 min"},
        {"l" + std::to_string(antihydra_graph_n) + "g2", "0"},
        {"l" + std::to_string(antihydra_graph_n) + "b2", "0"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Starting at 0, the probability of retiring is quite high, but most of that comes from retiring almost immediately, which we know Antihydra doesn't do."), 5);
    // substitute in 0 and adjust graph to just be one point at (0,0)
    antihydra_t = 0;
    antihydra_d = 49;
    for (int i=0; i<antihydra_graph_n; i++) {
        as->manager.transition(MICRO, {
            {"p" + std::to_string(i+4) + "y", std::to_string(antihydra_t)},
        });
        antihydra_t += 2 - ((antihydra_d&3)<3) * 3;
        antihydra_d += antihydra_d >> 2;
    }
    as->manager.transition(MICRO, {
        {"gp", "0"},
        {"l" + std::to_string(antihydra_graph_n) + "g2", "1"},
        {"l" + std::to_string(antihydra_graph_n) + "b2", "1"},
    });
    ls->begin_latex_transition(MICRO, "P(0 \\to -1) = \\frac{1}{\\varphi^{0+1}}");
    cs.render_microblock();
    antihydra_t = 0;
    antihydra_d = 1;
    for (int i=0; i<8; i++) {
        as->manager.set({
            {"p" + std::to_string(i+4) + "y", std::to_string(antihydra_t)},
        });
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.transition(MICRO, {
        {"gp", "1"},
    });
    ls->begin_latex_transition(MICRO, "P(0 \\to -1) = \\frac{1}{\\varphi^{1}} \\approx 61.8\\%");
    cs.render_microblock();
    antihydra_t = 0;
    antihydra_d = 86;
    for (int i=0; i<8; i++) {
        as->manager.transition(MICRO, {
            {"p" + std::to_string(i+4) + "y", std::to_string(antihydra_t)},
        });
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.transition(MICRO, {
        {"gp", "4"},
    });
    cs.render_microblock();
    antihydra_t = 0;
    antihydra_d = 10;
    for (int i=0; i<8; i++) {
        as->manager.transition(MICRO, {
            {"p" + std::to_string(i+4) + "y", std::to_string(antihydra_t)},
        });
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.transition(MICRO, {
        {"gp", "7"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"gp", "0"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Researchers have simulated Antihydra for a long time, and after all those simulations, the edge thickness is over 137 billion, which makes the retiring probability less than 1 in 10^(28 billion). Basically zero."), 2);
    cs.manager.set("ls.opacity", "0");
    as->reset();
    // extend graph a lot
    as->add_point();
    as->add_point({0});
    as->add_point({0});
    as->add_point({0});
    as->add_line(0, 2);
    as->add_line(0, 3);
    as->manager.set({
        {"p0x", "0.1"},
        {"p0y", "0.7"},
        {"p0o", "0"},
        {"p1x", "0.8 <gp> 2 + /"},
        {"p1y", "<p1x> -0.8 /"},
        {"p1o", "0"},
        {"p2x", "0.8"},
        {"p2o", "0"},
        {"p3y", "-0.6"},
        {"p3o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "s", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "s", "1"},
        {"gp", "e <gzoom> -1 * ^ 1 -"},
        {"gzoom", "0"},
    });
    antihydra_t = 0;
    antihydra_d = 8;
    antihydra_graph_n = 100;
    for (int i=0; i<antihydra_graph_n; i++) {
        as->add_point();
        if (i>0) as->add_line(as->num_points-2, as->num_points-1);
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "xeq", "<p0xeq> <p1x> <p" + std::to_string(as->num_points-1) + "x> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "xeq> " + std::to_string(i) + " <gp> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "yeq", "<p0yeq> <p1y> <p" + std::to_string(as->num_points-1) + "y> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "yeq> " + std::to_string(i) + " <gp> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "x", std::to_string(i)},
            {"p" + std::to_string(as->num_points-1) + "y", std::to_string(antihydra_t)},
            {"p" + std::to_string(as->num_points-1) + "o", "0"},
        });
        /*if (i>0) as->manager.set({
            {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "0"},
        });
        if (i>0) as->manager.transition(MICRO, {
            {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "1"},
        });*/
        antihydra_t += 2 - (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.transition(MICRO, {
        {"p1x", "0.8 <gp> /"},
        {"gp", "e <gzoom> -1 * ^ 1 +"},
        {"gzoom", std::to_string(antihydra_graph_n) + " log -1 *"},
    });
    // raise the substituted t_0 to 137000000000
    as->add_point();
    as->add_text(as->num_points-1, {"P(", " \\to -1) = \\frac{1}{\\varphi^{", "}} \\approx ", "\\%"});
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "x", "0.5"},
        {"p" + std::to_string(as->num_points-1) + "y", "0.85"},
        {"t0h", "0.2"},
        {"t0v0", "<t0v1> 1 -"},
        {"t0v1", "e <t0vlog> ^"},
        {"t0v2", "100 phi <t0vlog> 3 * ^ /"},
        {"t0vlog", "0"},
    });
    as->manager.transition(MICRO, {
        {"t0vlog", "136985635424 log"},
    });
    as->add_text(4, {"t_0"});
    as->manager.set({
        {"t1x", "-0.02"},
    });
    as->manager.transition(MICRO, {
        {"t1o", "0"},
    });
    cs.render_microblock();
    cs.manager.set("ls.opacity", "1");
    as->manager.set({
        {"t0o", "0"},
    });
    ls->jump_latex("P(137000000000 \\to -1) = \\frac{1}{\\varphi^{137000000001}} \\approx 0\\%");
    ls->begin_latex_transition(MICRO, "P(137000000000 \\to -1) \\approx 0%");
    cs.render_microblock();

    stage_macroblock(FileBlock("We can take an educated guess that Antihydra doesn't retire."), 2);
    // the "P \approx 0" centers as everything else fades out
    ls->begin_latex_transition(MICRO, "P \\approx 0\\%");
    cs.fade_subscene(MICRO, "as", 0);
    cs.manager.transition(MICRO, "ls.y", "0.5");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(0.5), 1);
    cs.fade_subscene(MICRO, "ls", 0);
    cs.render_microblock();
    as->reset();


    stage_macroblock(FileBlock("Soon after Antihydra, other cryptids with 6 emotions started being discovered."), 3);
    // antihydra spacetime fades in
    cs.move_to_back("as");
    cs.fade_subscene(MICRO, "as", 1);
    cs.fade_subscene(MICRO, "tms0", 1);
    tms0->default_everything(true, true);
    tms1->default_everything(true, true);
    tms2->default_everything(true, true);
    tms3->default_everything(true, true);
    parse_tmstring(tmstrs[space_needle], 6, 2, tm);
    tms1->set_tm(tm);
    parse_tmstring(tmstrs[lucy], 6, 2, tm);
    tms2->set_tm(tm);
    for (int i=0; i<4; i++) {
        cs.manager.set({
            {"tms" + std::to_string(i) + ".w", "0.2"},
            {"tms" + std::to_string(i) + ".h", "0.6"},
            {"tms" + std::to_string(i) + ".zoom", "-3"},
            {"tms" + std::to_string(i) + ".stfx", "0"},
            {"tms" + std::to_string(i) + ".stfy", "25"},
            {"tms" + std::to_string(i) + ".vs", "0.4"},
        });
        tms[i]->manager.set({
            {"w", "[tms" + std::to_string(i) + ".w]"},
            {"h", "[tms" + std::to_string(i) + ".h]"},
            {"zoom", "[tms" + std::to_string(i) + ".zoom]"},
            {"center_x", "[tms" + std::to_string(i) + ".stfx]"},
            {"spacetime_focus_y", "[tms" + std::to_string(i) + ".stfy]"},
            {"vertical_step", "[tms" + std::to_string(i) + ".vs]"},
            {"iterations", "200"},
        });
    }
    cs.manager.set({
        {"tms0.w", "0.6"},
        {"tms1.x", "1.2"},
        {"tms1.opacity", "1"},
        {"tms2.x", "1.5"},
        {"tms2.opacity", "1"},
        {"tms3.opacity", "0"},
    });
    tms3->manager.set({
        {"iterations", "10000"},
    });
    as->reset();
    as->add_point();
    as->add_point({0});
    as->add_point();
    as->add_point({2});
    as->add_point();
    as->add_point({4});
    add_rect(as, 0, 1, labels[antihydra], 0);
    add_rect(as, 2, 3, labels[space_needle], 1);
    add_rect(as, 4, 5, labels[lucy], 2);
    cs.render_microblock();
    // space needle and lucy slide into view
    cs.manager.transition(MICRO, {
        {"tms0.x", "0.2"},
        {"tms0.w", "0.2"},
        {"tms1.x", "0.5"},
        {"tms2.x", "0.8"},
    });
    cs.render_microblock();
    cs.fade_subscene(MICRO, "tms1", 0);
    cs.render_microblock();

    stage_macroblock(FileBlock("They lie on a spectrum of how sure we are that they're unsolvable."), 2);
    // spectrum shows up
    spect_start = as->num_points;
    ln = as->line_endpoints.size();
    create_spectrum(as, true);
    as->manager.transition(MICRO, {
        {"t0y", "<t0yabs> -1 *"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.y", "0.8"},
        {"tms0.h", "16 45 /"},
        {"tms2.y", "0.2"},
        {"tms2.h", "16 45 /"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("On one end of the spectrum we have the cryptids that are unpredictable because they multiply by fractions and then use remainders as coin flips, which are called Collatz-like."), 4);
    cs.manager.transition(MICRO, {
        {"tms0.x", "0.3"},
        {"tms2.x", "0.3"},
    });
    cs.render_microblock();
    // connect antihydra & lucy to the spectrum
    connect_to_spectrum(as, spect_start, 0, spect_pos[antihydra]);
    connect_to_spectrum(as, spect_start, 4, spect_pos[lucy]);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("On the other end, we have some beavers who seem chaotic, but have a sort of buffer zone preventing them from retiring, and this buffer zone empirically seems to mostly grow."), 5);
    cs.move_to_front("tms3");
    cs.fade_subscene(MICRO, "tms3", 1);
    parse_tmstring(tmstrs[chaos_with_buffer], 6, 2, tm);
    tms3->set_tm(tm);
    // add chaos with buffer to the spectrum, then show the spacetime diagram
    pn = as->num_points;
    ln = as->line_endpoints.size();
    as->add_point();
    as->add_point({pn});
    add_rect(as, pn, pn+1, labels[chaos_with_buffer], 3);
    connect_to_spectrum(as, spect_start, pn, spect_pos[chaos_with_buffer]);
    cs.manager.set({
        {"tms3.x", "0.7"},
        {"tms3.y", "0.2"},
        {"tms3.w", "0.2"},
        {"tms3.h", "16 45 /"},
    });
    //as->manager.set("p" + std::to_string(as->num_points-2) + "xeq", "<p" + std::to_string(pn) + "xeq>");
    as->manager.transition(MICRO, {
        {"r" + std::to_string(pn) + "o", "1"},
    });
    cs.render_microblock();
    // expand the spacetime diagram to cover the whole screen
    cs.manager.transition(MICRO, {
        {"tms3.x", "0.5"},
        {"tms3.y", "0.5"},
        {"tms3.w", "1"},
        {"tms3.h", "1"},
    });
    cs.fade_subscene(MICRO, "tms0", 0);
    cs.fade_subscene(MICRO, "tms2", 0);
    for (int i=0; i<as->num_points; i++) as->manager.transition(MICRO, "p" + std::to_string(i) + "o", as->manager.get_equation_string("p" + std::to_string(i) + "o") + " 0 *");
    for (int i=0; i<as->line_endpoints.size(); i++) as->manager.transition(MICRO, "l" + std::to_string(i) + "o", as->manager.get_equation_string("l" + std::to_string(i) + "o") + " " + std::to_string(1-(i>ln+3 || i<ln)) + " *");
    for (int i=0; i<as->text_anchors.size(); i++) as->manager.transition(MICRO, "t" + std::to_string(i) + "o", as->manager.get_equation_string("t" + std::to_string(i) + "o") + " 0 *");
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms3.vs", "0.2"},
        {"tms3.zoom", "-4"},
        {"tms3.stfx", "3"},
        {"tms3.stfy", "1350"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("We don't understand the chaos, so maybe there is some way for us to prove the buffer zone will stay there forever, but that would be surprising."), 1);
    cs.manager.transition(MICRO, {
        {"tms3.zoom", "-4.5"},
        {"tms3.stfy", "7500"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("The existence of this buffer zone allows the remaining chaos to be chaos."), 2);
    // show CPS, CTL, FAR failing to decide it, maybe?
    cs.manager.transition(MICRO, {
        {"tms3.zoom", "-5.5"},
        {"tms3.stfy", "5000"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms3.vs", "0.4"},
        {"tms3.zoom", "-3"},
        {"tms3.stfx", "0"},
        {"tms3.stfy", "25"},
    });
    cs.render_microblock();

    // leave this out?
    //stage_macroblock(FileBlock("It's no longer likely to have a pattern hidden in it that would prevent the beaver from retiring, because the buffer zone is enough to explain why the beaver hasn't retired by pure chance - it wasn't given many chances."), 1);
    // highlight the "chances" - moments when the buffer was expended
    //cs.render_microblock();

    stage_macroblock(FileBlock("In between those extremes lie beavers like Space Needle, and the currently unnamed beaver who inspired the Beaver Math Olympiad, a collection of problems stated in the style of math olympiads, but the problems all come from studying beavers."), 6);
    // unexpand the spacetime diagram to reveal the rest of the spectrum again, add Space Needle and BMO1 to the spectrum, and show a scrolling BMO wiki page
    cs.manager.transition(MICRO, {
        {"tms3.x", "0.7"},
        {"tms3.y", "0.2"},
        {"tms3.w", "0.2"},
        {"tms3.h", "16 45 /"},
    });
    cs.fade_subscene(MICRO, "tms0", 1);
    for (int i=0; i<as->num_points; i++) {
        string s = as->manager.get_equation_string("p" + std::to_string(i) + "o");
        as->manager.transition(MICRO, "p" + std::to_string(i) + "o", s.substr(0, s.find_last_of(" ",s.find_last_of(" ")-1)));
    }
    for (int i=0; i<as->line_endpoints.size(); i++) {
        string s = as->manager.get_equation_string("l" + std::to_string(i) + "o");
        if (!(8<=i && i<12) && i!=ln-1) as->manager.transition(MICRO, "l" + std::to_string(i) + "o", s.substr(0, s.find_last_of(" ",s.find_last_of(" ")-1)));
    }
    for (int i=0; i<as->text_anchors.size(); i++) {
        string s = as->manager.get_equation_string("t" + std::to_string(i) + "o");
        if (i!=2) as->manager.transition(MICRO, "t" + std::to_string(i) + "o", s.substr(0, s.find_last_of(" ",s.find_last_of(" ")-1)));
    }
    cs.render_microblock();
    pn = as->num_points;
    ln = as->line_endpoints.size();
    as->add_point();
    as->add_point({pn});
    add_rect(as, pn, pn+1, labels[space_needle]);
    connect_to_spectrum(as, spect_start, pn, spect_pos[space_needle]);
    as->manager.set({
        {"p" + std::to_string(pn  ) + "x", "[tms1.x]"},
        {"p" + std::to_string(pn  ) + "y", "[tms1.y]"},
        {"p" + std::to_string(pn+1) + "x", "[tms1.w] 2 /"},
        {"p" + std::to_string(pn+1) + "y", "[tms1.h] 2 /"},
        {"r" + std::to_string(pn) + "o", "0"},
    });
    cs.manager.set({
        {"tms1.x", "0.3"},
        {"tms1.y", "0.2"},
        {"tms1.w", "0.2"},
        {"tms1.h", "16 45 /"},
    });
    cs.fade_subscene(MICRO, "tms1", 1);
    as->manager.transition(MICRO, {
        {"r" + std::to_string(pn) + "o", "1"},
    });
    cs.render_microblock();
    parse_tmstring(tmstrs[bmo1], 6, 2, tm);
    tms2->set_tm(tm);
    pn = as->num_points;
    ln = as->line_endpoints.size();
    tn = as->text_anchors.size();
    as->add_point();
    as->add_point({pn});
    add_rect(as, pn, pn+1, labels[bmo1]);
    connect_to_spectrum(as, spect_start, pn, spect_pos[bmo1]);
    as->manager.set({
        {"p" + std::to_string(pn  ) + "x", "[tms2.x]"},
        {"p" + std::to_string(pn  ) + "y", "[tms2.y]"},
        {"p" + std::to_string(pn+1) + "x", "[tms2.w] 2 /"},
        {"p" + std::to_string(pn+1) + "y", "[tms2.h] -2 /"},
        {"r" + std::to_string(pn) + "o", "0"},
        {"t" + std::to_string(tn) + "y", "<t" + std::to_string(tn) + "yabs> -1 *"},
    });
    cs.manager.set({
        {"tms2.x", "0.7"},
        {"tms2.y", "0.8"},
        {"tms2.w", "0.2"},
        {"tms2.h", "16 45 /"},
    });
    cs.fade_subscene(MICRO, "tms2", 1);
    as->manager.transition(MICRO, {
        {"r" + std::to_string(pn) + "o", "1"},
    });
    cs.render_microblock();
    // TODO: BMO wiki page screenshots?
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "as", 0);
    cs.fade_subscene(MICRO, "tms0", 0);
    cs.fade_subscene(MICRO, "tms1", 0);
    cs.fade_subscene(MICRO, "tms2", 0);
    cs.fade_subscene(MICRO, "tms3", 0);
    cs.render_microblock();


    stage_macroblock(FileBlock("All of these cryptids are expected not to retire."), 2);
    cs.fade_subscene(MICRO, "as", 1);
    as->reset();
    create_spectrum(as);
    as->add_point({2});
    as->add_point({2});
    as->add_line(2, 3);
    as->add_line(2, 4);
    as->add_text(3, {"P=100\\%"});
    as->add_text(4, {"P=0\\%"});
    as->manager.set({
        {"t0x", "0.06"},
        {"t0o", "0"},
        {"t0h", "0.05"},
        {"t1x", "-0.045"},
        {"t1o", "0"},
        {"t1h", "0.05"},
    });
    vector<float> xs = {0.1, 0.1+11.0/80, 0.5, 0.9-11.0/80, 0.9};
    for (int i=0; i<xs.size(); i++) {
        as->add_point();
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "x", "<p0xeq> <p1xeq> " + std::to_string(xs[i]) + " lerp"},
            {"p" + std::to_string(as->num_points-1) + "y", "<p4yeq>"},
            //{"p" + std::to_string(as->num_points-1) + "b", "0"},
        });
    }
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"p3y", "-0.25 0.5625 /"},
        {"p4y", "0.25 0.5625 /"},
        {"l2r2", "0"},
        {"l2b2", "0"},
        {"l3g2", "0"},
        {"t0o", "1"},
        {"t1o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("And that makes sense, because if they were likely to retire, then they would be likely to retire early enough for us to see it."), 1);
    as->add_point();
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "x", "<p0xeq> <p1xeq> 0.1 lerp"},
        {"p" + std::to_string(as->num_points-1) + "y", "<p3yeq>"},
        {"p" + std::to_string(as->num_points-1) + "o", "0"},
        //{"p" + std::to_string(as->num_points-1) + "b", "0"},
    });
    as->manager.transition(MICRO, {
        {"p" + std::to_string(as->num_points-1) + "o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("If we saw them retire, then they would be solved, and therefore not cryptids."), 1);
    as->manager.set({
        {"p" + std::to_string(as->num_points-1) + "o", "1 {microblock_fraction} -"},
    });
    as->manager.transition(MICRO, {
        {"p" + std::to_string(as->num_points-1) + "o", "0"},
        {"p" + std::to_string(as->num_points-1) + "t", "0.1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("But are there any exceptions?"), 2);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();
    // TNF zoom to lucy
    cs.render_microblock();


    stage_macroblock(FileBlock("Meet Lucy. She really wants to see an eclipse, and while building her dams, she often checks if an eclipse is happening."), 5);
    // TODO: make lucy animations better, this shits boring
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_text(0, {"\\text{Day ", "}"});
    parse_tmstring(tmstrs[lucy], 6, 2, tm);
    tms0->set_tm(tm);
    cs.manager.set({
        {"tms0.x", "0.5"},
        {"tms0.y", "0.6"},
        {"tms0.w", "1"},
        {"tms0.h", "0.8"},
        {"tms0.iterations", "0"},
        {"tms0.zoom", "-2.5"},
        {"tms0.stfx", "0"},
        {"tms0.stfy", "<tms0.iterations> 6 -"},
    });
    cs.fade_subscene(MICRO, "tms0", 1);
    as->manager.set({
        {"p0x", "0.5"},
        {"p0y", "0.1"},
        {"t0h", "0.2"},
        {"t0v0", "[tms0.iterations] floor"},
    });
    tms0->manager.set({
        {"iterations", "[tms0.iterations]"},
        {"opacity_min", "0.4"},
        {"current_tape_opacity", "1"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "54"},
    });
    cs.render_microblock();
    // TODO: thought bubble saying "oh it's coming up soon! i'll just build this little bit..."
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "236"},
        {"tms0.zoom", "-3"},
        {"tms0.stfx", "15"},
    });
    cs.render_microblock();
    // TODO: thought bubble saying "dang, i missed it!"
    cs.render_microblock();

    stage_macroblock(FileBlock("If she gets to see her creation in the light of an eclipse, she will retire, but until then, she will keep building dams, to make the view during the eclipse as beautiful as she can."), 1);
    as->text_latex[0] = {"\\text{Maybe someday...}"};
    tms0->reset({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 0, 30);
    cs.manager.set({
        {"tms0.iterations", "970"},
        {"tms0.stfx", "{microblock_fraction} 60 * 30 -"},
        {"tms0.stfy", "<tms0.iterations> 13 -"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "1046"},
    }, false);
    cs.render_microblock();
    cs.manager.set({
        {"tms0.stfx", "30"},
    });

    stage_macroblock(FileBlock("Each time there's an eclipse, she has a 1 in 5 chance of seeing it, but she missed the first three eclipses."), 1);
    // third one at day 445256195, start from 0^inf (10)^a A> (10)^b 0^inf, thought bubble saying "dang, i missed it!" again
    as->text_latex[0] = {"\\text{Day 445256", "}"};
    tms0->reset({1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 0, 2);
    cs.manager.set({
        {"tms0.zoom", "-2.3"},
        {"tms0.stfx", "0"},
        {"tms0.stfy", "<tms0.iterations> 5 -"},
        {"tms0.iterations", "20"},
    });
    as->manager.set({
        {"t0v0", "[tms0.iterations] floor 137 +"}
    });
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "58"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(0.5), 1);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();

    stage_macroblock(FileBlock("While we can predict the next eclipse based on what happened before the last one, predicting when exactly Lucy will check for it is difficult, because Lucy computes a Collatz-like sequence, just like Antihydra."), 4);
    cs.fade_subscene(MICRO, "tms0", 0);
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_point();
    as->add_point({0,1});
    as->add_line(0,1);
    int ecl_timeline_n = 9;
    for (int i=0; i<ecl_timeline_n; i++) {
        as->add_point({0,1});
        as->add_icon(i+3);
        string d = std::to_string(1.65 * i + 0.2 + (1.618*(i+1)*(i+1) - floor(1.618*(i+1)*(i+1))));
        as->manager.set({
            {"p"+std::to_string(i+3)+"w1", "<tltime> <tltime> " + d + " - floor - <tltime> " + d + " - <tltime> " + d + " - floor - 4 * 3 - 0 max -"},
            {"p"+std::to_string(i+3)+"w0", "15 <p"+std::to_string(i+3)+"w1> -"},
            {"i"+std::to_string(i)+"o", "0"},
            {"i"+std::to_string(i)+"h", "0.05"},
            {"i"+std::to_string(i)+"id", std::to_string(icons_whnm.w+4)},
        });
    }
    global_t = get_global_state("t");
    as->manager.set({
        {"p0x", "0.5"},
        {"p0y", "0.5"},
        {"p1x", "0.5"},
        {"p1y", "0.5"},
        {"p0o", "0"},
        {"p1o", "0"},
        {"p2o", "0"},
        {"tltime", "{t} " + std::to_string(global_t) + " - 0.7 *"},
    });
    as->manager.transition(MICRO, {
        {"p0x", "-0.1"},
        {"p1x", "1.1"},
        {"p2o", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    for (int i=0; i<ecl_timeline_n; i++) {
        as->manager.transition(MICRO, "i"+std::to_string(i)+"o", "1");
    }
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("In our world, the intervals between eclipses are all equally long, around half a year."), 3);
    // show timeline of irl eclipses, with the interval lengths written
    for (int i=0; i<ecl_timeline_n; i++) {
        as->manager.transition(MICRO, "i"+std::to_string(i)+"o", "0");
    }
    as->manager.transition(MICRO, "p2o", "0");
    cs.render_microblock();
    as->add_text(2, {"\\text{20th April 2023}"}, false);
    as->add_text(3, {"\\text{14th October 2023}"}, false);
    as->add_text(4, {"\\text{8th April 2024}"}, false);
    as->add_text(5, {"\\text{2nd October 2024}"}, false);
    for (int i=0; i<ecl_timeline_n; i++) {
        if (i>0) {
            as->add_point({i+1,i+2});
            as->add_text(ecl_timeline_n+i+2, {strmul("\\approx", i>3) + strmul("10^{", max(0,i-3)), strmul("}", max(0,i-3)) + "\\text{ months}"});
            as->manager.set({
                {"t"+std::to_string(i+3)+"y", "0.05"},
                {"t"+std::to_string(i+3)+"h", "0.05"},
                {"t"+std::to_string(i+3)+"o", "0"},
                {"t"+std::to_string(i+3)+"v0", strmul("e ", i>2) + "<t"+std::to_string(i+3)+"v>" + strmul(" ^", i>2) + " floor"},
                {"t"+std::to_string(i+3)+"v", "6" + strmul(" log", i>2)},
            });
            as->manager.transition(MICRO, {
                {"t"+std::to_string(i+3)+"o", "1"},
            });
        }
        as->manager.set({
            {"p"+std::to_string(i+2)+"o", "0"},
            {"p"+std::to_string(i+2)+"w0", std::to_string(ecl_timeline_n-i)},
            {"p"+std::to_string(i+2)+"w1", std::to_string(i+1)},
        });
        as->manager.transition(MICRO, {
            {"p"+std::to_string(i+2)+"o", "1"},
        });
    }
    as->manager.set({
        {"p0x", "-0.2"},
        {"p1x", std::to_string(0.28*(ecl_timeline_n+1) - 0.2)},
    });
    for (int i=0; i<4; i++) {
        as->manager.set({
            {"t"+std::to_string(i)+"y", "-0.05"},
            {"t"+std::to_string(i)+"h", "0.05"},
            {"t"+std::to_string(i)+"o", "0"},
        });
        as->manager.transition(MICRO, {
            {"t"+std::to_string(i)+"o", "1"},
        });
    }
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("In Lucy's world, eclipses are much more rare - each interval's length is exponential in the previous interval's length."), 3);
    // replace irl eclipse interval lengths with those from lucy's world, and show the exponential approximations
    as->manager.transition(MICRO, {
        {"t0o", "0"},
        {"t1o", "0"},
        {"t2o", "0"},
        {"t3o", "0"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"t4v", "2"},
        {"t5v", "14"},
        {"t6v", "11292 log"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This is why we've only been able to see that Lucy missed the first few eclipses. The date of the next one has 2900 digits!"), 1);
    // extend timeline rightwards to show next few eclipses
    as->manager.transition(MICRO, {
        {"p0x", "-0.25"},
        {"p1x", std::to_string(0.25*(ecl_timeline_n+1) - 0.25)},
        {"t7v", "2902.5 log"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("However, since the probability of missing all infinitely many 20% chances is 0, Lucy's probability of eventually retiring is 100%."), 1);
    as->manager.transition(MICRO, {
        {"p0x", std::to_string(-1.0f/(ecl_timeline_n+1))},
        {"p1x", std::to_string(1+1.0f/(ecl_timeline_n+1))},
    });
    for (int i=5; i<ecl_timeline_n; i++) {
        as->manager.transition(MICRO, {
            {"t"+std::to_string(i+3)+"v", "2900.5 log"},
        });
    }
    // TODO: show an X or 0% or something above the ones we've seen her not retire at, and then probability of retiring above all the others
    cs.render_microblock();


    stage_macroblock(FileBlock("There are a few other beavers like Lucy. They compute a Collatz-like random walk that always decreases."), 3);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();
    cs.fade_subscene(MICRO, "as", 1);
    // show graph of lucy walk
    as->add_point();
    as->add_point({0});
    as->add_point({0});
    as->add_point({0});
    as->add_line(0, 2);
    as->add_line(0, 3);
    as->manager.set({
        {"p0x", "0.15"},
        {"p0y", "0.92"},
        {"p0o", "0"},
        {"p1x", "0.7"},
        {"p1y", "<p1x> -0.72 *"},
        {"p1o", "0"},
        {"p2x", "<p1x>"},
        {"p2o", "0"},
        {"p3y", "<p1x> -1.2 *"},
        {"p3o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "s", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "s", "1"},
        {"gp", "0"},
    });
    cs.render_microblock();
    int lucy_t = 14;
    int lucy_d = 4;
    int lucy_graph_n = 9;
    int lucy_const_coefs[3] = {3, 11, 12};
    for (int i=0; i<lucy_graph_n; i++) {
        as->add_point();
        if (i>0) as->add_line(as->num_points-2, as->num_points-1);
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "xeq", "<p0xeq> <p1x> " + std::to_string(lucy_graph_n) + " / <p" + std::to_string(as->num_points-1) + "x> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "xeq> " + std::to_string(i) + " <gp> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "yeq", "<p0yeq> <p1y> " + std::to_string(lucy_graph_n) + " / <p" + std::to_string(as->num_points-1) + "y> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "yeq> " + std::to_string(i) + " <gp> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "x", std::to_string(i)},
            {"p" + std::to_string(as->num_points-1) + "y", std::to_string(lucy_t)},
            //{"p" + std::to_string(as->num_points-1) + "o", "0"},
        });
        if (i>0) {
            as->manager.set({
                {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "0"},
            });
            as->manager.transition(MICRO, {
                {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "1"},
            });
        }
        lucy_t -= 1 + (lucy_d%3>0);
        lucy_d = 8*(lucy_d/3) + lucy_const_coefs[lucy_d%3];
    }
    as->manager.transition(MICRO, {
        {"gp", std::to_string(lucy_graph_n-1)},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Every time it drops to 0, they either retire, or they reset it to the large number that they've been repeatedly multiplying to get their coin flips."), 2);
    // shrink graph and show retirement option
    as->manager.transition(MICRO, {
        {"p0x", "0.04"},
        {"p0y", "0.74"},
        {"p1x", "0.4"},
    });
    pn = as->num_points;
    ln = as->line_endpoints.size();
    as->add_point();
    as->add_point({pn});
    as->add_point({pn});
    as->add_point();
    as->add_line(pn, pn+1);
    as->add_line(pn, pn+2);
    as->add_icon(pn+3);
    as->manager.set({
        {"p" + std::to_string(pn  ) + "x", "0.47"},
        {"p" + std::to_string(pn  ) + "y", "0.5"},
        {"p" + std::to_string(pn  ) + "o", "0"},
        {"p" + std::to_string(pn+1) + "x", "1 2 <p" + std::to_string(pn) + "xeq> * -"},
        {"p" + std::to_string(pn+1) + "y", "<p" + std::to_string(pn+1) + "x> -1 *"},
        {"p" + std::to_string(pn+1) + "o", "0"},
        {"p" + std::to_string(pn+2) + "x", "<p" + std::to_string(pn+1) + "x>"},
        {"p" + std::to_string(pn+2) + "y", "<p" + std::to_string(pn+1) + "y> -1 *"},
        {"p" + std::to_string(pn+2) + "o", "0"},
        {"p" + std::to_string(pn+3) + "x", "<p" + std::to_string(pn) + "xeq> <p" + std::to_string(pn+1) + "x> 3 * +"},
        {"p" + std::to_string(pn+3) + "y", "<p" + std::to_string(pn) + "yeq> <p" + std::to_string(pn+1) + "y> 4 * +"},
        {"p" + std::to_string(pn+3) + "o", "0"},
        {"l" + std::to_string(ln  ) + "y", "<p" + std::to_string(pn+1) + "y> 0.5 *"},
        {"l" + std::to_string(ln  ) + "o", "0"},
        {"l" + std::to_string(ln  ) + "s", "1"},
        {"l" + std::to_string(ln+1) + "y", "<p" + std::to_string(pn+2) + "y> 0.5 *"},
        {"l" + std::to_string(ln+1) + "o", "0"},
        {"l" + std::to_string(ln+1) + "s", "1"},
        {"i" + std::to_string(as->icon_anchors.size()-1) + "o", "0"},
        {"i" + std::to_string(as->icon_anchors.size()-1) + "h", "0.33"},
        {"i" + std::to_string(as->icon_anchors.size()-1) + "id", std::to_string(icons_len-1)},
    });
    as->manager.transition(MICRO, {
        {"l" + std::to_string(ln  ) + "o", "1"},
        {"i" + std::to_string(as->icon_anchors.size()-1) + "o", "1"},
    });
    cs.render_microblock();
    // show reset option, which is another graph but starting from higher up
    pn = as->num_points;
    as->add_point();
    as->add_point({pn});
    as->add_point({pn});
    as->add_point({pn});
    as->add_line(pn, pn+2);
    as->add_line(pn, pn+3);
    as->manager.set({
        {"p" + std::to_string(pn  ) + "x", "0.56"},
        {"p" + std::to_string(pn  ) + "y", "0.95"},
        {"p" + std::to_string(pn  ) + "o", "0"},
        {"p" + std::to_string(pn+1) + "x", "0.4"},
        {"p" + std::to_string(pn+1) + "y", "<p" + std::to_string(pn+1) + "x> -0.8 *"},
        {"p" + std::to_string(pn+1) + "o", "0"},
        {"p" + std::to_string(pn+2) + "x", "<p" + std::to_string(pn+1) + "x>"},
        {"p" + std::to_string(pn+2) + "o", "0"},
        {"p" + std::to_string(pn+3) + "y", "<p" + std::to_string(pn+1) + "x> -1.2 *"},
        {"p" + std::to_string(pn+3) + "o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "s", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "s", "1"},
        {"gp2", "0"},
    });
    lucy_t = 130;
    lucy_d = 4;
    lucy_graph_n = 90;
    for (int i=0; i<lucy_graph_n; i++) {
        as->add_point();
        if (i>0) as->add_line(as->num_points-2, as->num_points-1);
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "xeq", "<p" + std::to_string(pn) + "xeq> <p" + std::to_string(pn+1) + "x> " + std::to_string(lucy_graph_n) + " / <p" + std::to_string(as->num_points-1) + "x> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "xeq> " + std::to_string(i) + " <gp2> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "yeq", "<p" + std::to_string(pn) + "yeq> <p" + std::to_string(pn+1) + "y> " + std::to_string(lucy_graph_n) + " / <p" + std::to_string(as->num_points-1) + "y> * +" + (i>0 ? " <p" + std::to_string(as->num_points-2) + "yeq> " + std::to_string(i) + " <gp2> - 0 max 1 min lerp" : "")},
            {"p" + std::to_string(as->num_points-1) + "x", std::to_string(i)},
            {"p" + std::to_string(as->num_points-1) + "y", std::to_string(lucy_t)},
            {"p" + std::to_string(as->num_points-1) + "o", "0"},
        });
        if (i>0) {
            as->manager.set({
                {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "0"},
            });
            as->manager.transition(MICRO, {
                {"l" + std::to_string(as->line_endpoints.size()-1) + "o", "1"},
            });
        }
        lucy_t -= 1 + (lucy_d%3>0);
        lucy_d = 8*(lucy_d/3) + lucy_const_coefs[lucy_d%3];
    }
    as->manager.transition(MICRO, {
        {"gp2", std::to_string(lucy_graph_n-1)},
        {"l" + std::to_string(ln+1) + "o", "1"},
    });
    cs.render_microblock();

    //stage_macroblock(FileBlock("Whether they retire is based on whether the random walk skips 0 and goes into the negatives (which also works like a coin flip), and sometimes one more coin flip."), 1);
    stage_macroblock(FileBlock("Whether they retire is based on more coin flips."), 1);
    // TODO: something
    cs.render_microblock();

    stage_macroblock(SilenceBlock(0.5), 1);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();

    stage_macroblock(FileBlock("They're called eclipse machines, and they are the most common type of cryptids that are expected to retire."), 3);
    cs.fade_subscene(MICRO, "as", 1);
    // back to cryptid spectrum
    spect_start = as->num_points;
    create_spectrum(as);
    shown_tms = {antihydra, space_needle, bmo1};
    vector<float> x_positions = {0.3, 0.3 + 11.0/80, 0.7};
    for (int i=0; i<3; i++) {
        pn = as->num_points;
        tn = as->text_anchors.size();
        as->add_point();
        as->add_point({pn});
        add_rect(as, pn, pn+1, labels[shown_tms[i]], i);
        parse_tmstring(tmstrs[shown_tms[i]], 6, 2, tm);
        tms[i]->set_tm(tm);
        cs.fade_subscene(MICRO, "tms" + std::to_string(i), 1);
        connect_to_spectrum(as, spect_start, pn, spect_pos[shown_tms[i]], 0.5, false);
        cs.manager.set({
            {"tms" + std::to_string(i) + ".x", std::to_string(x_positions[i])},
            {"tms" + std::to_string(i) + ".y", std::to_string(0.8 - 0.6*(i==1))},
            {"tms" + std::to_string(i) + ".w", "0.2"},
            {"tms" + std::to_string(i) + ".h", "16 45 /"},
            {"tms" + std::to_string(i) + ".iterations", "75"},
            {"tms" + std::to_string(i) + ".vs", "0.4"},
            {"tms" + std::to_string(i) + ".zoom", "-3"},
            {"tms" + std::to_string(i) + ".stfx", "0"},
            {"tms" + std::to_string(i) + ".stfy", "25"},
        });
        as->manager.set({
            {"t" + std::to_string(tn) + "y", "<t" + std::to_string(tn) + "yabs> " + std::to_string(2*(i==1) - 1) + " *"},
        });
    }
    cs.render_microblock();
    pn = as->num_points;
    as->add_point();
    as->add_point({pn});
    add_rect(as, pn, pn+1, "\\text{Eclipse Machines}", 3);
    parse_tmstring(tmstrs[lucy], 6, 2, tm);
    tms3->set_tm(tm);
    cs.fade_subscene(MICRO, "tms3", 1);
    connect_to_spectrum(as, spect_start, pn, 0.1);
    cs.manager.set({
        {"tms3.x", "0.3 11 160 / -"},
        {"tms3.y", "0.2"},
        {"tms3.w", "0.2"},
        {"tms3.h", "16 45 /"},
        {"tms3.iterations", "75"},
        {"tms3.vs", "0.4"},
        {"tms3.zoom", "-3"},
        {"tms3.stfx", "0"},
        {"tms3.stfy", "25"},
    });
    as->manager.transition(MICRO, {
        {"t" + std::to_string(as->text_anchors.size()-3) + "h", "0.09"},
        {"t" + std::to_string(as->text_anchors.size()-1) + "h", "0.09"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("For all we know, they might still avoid retirement forever, because they aren't truly random, and even if they were, probability 0 doesn't mean that it's impossible."), 1);
    // coin flipping heads over and over and over again
    cs.fade_subscene(MICRO, "tms0", -1);
    cs.fade_subscene(MICRO, "tms2", -1);
    int coins_n = 20;
    global_t = get_global_state("t");
    for (int i=0; i<coins_n; i++) {
        as->add_point();
        add_coin(as, as->num_points-1);
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "x", std::to_string(i * 1.0f/(coins_n-1))},
            {"p" + std::to_string(as->num_points-1) + "y", "0.8"},
            {"c" + std::to_string(as->num_points-1) + "o", "<cfp> " + std::to_string(0.1f*i) + " -"},
            {"c" + std::to_string(as->num_points-1) + "r", "0.045"},
            {"c" + std::to_string(as->num_points-1) + "a", "{t} " + std::to_string(global_t+6+0.2f*i) + " - 0 min " + std::to_string(15+0.2f*i) + " *"},
        });
    }
    as->manager.set({
        {"cfp", "0"},
    });
    as->manager.transition(MICRO, {
        {"cfp", "3"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("But we're preeeetty sure they do eventually retire, they just take so long that we don't get to see it, and they get to keep their cryptid status."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "as", 0);
    for (int i=0; i<4; i++) cs.fade_subscene(MICRO, "tms" + std::to_string(i), 0);
    cs.render_microblock();
    as->reset();

    // leave this one out?
    /*
    stage_macroblock(FileBlock("With one exception, that is. A cryptid that, after seeing an eclipse, doesn't necessarily retire."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Instead, it has a 40% chance to retire, and a 60% chance to start moving periodically forever."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("For this eclipse machine, even our best guess is not so black-and-white."), 1);
    cs.render_microblock();
    */


    stage_macroblock(FileBlock("You might be wondering: Why are they called eclipse *machines*?"), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Well, the beavers we've been talking about are actually what mathematicians call Turing machines."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("They were discovered by Alan Turing, the father of computer science, and they are the original inspiration for how modern computers work."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("For any program you can run on your computer, there's a beaver out there doing the exact same thing on a river!"), 1);
    cs.render_microblock();


    stage_macroblock(FileBlock("So far, all beavers we talked about could only build two types of dams (counting 'no dam' as a type of dam)."), 2);
    parse_tmstring("0RB1R-1R-1R-1R-_1RC1R-1R-1R-1R-_2RD1R-1R-1R-1R-_3RE1R-1R-1R-1R-_4RA1R-1R-1R-1R-", 5, 5, tm);
    tms0->set_tm(tm);
    cs.fade_subscene(MICRO, "tms0", 1);
    tms0->default_everything(true);
    cs.manager.set({
        {"tms0.x", "0.5"},
        {"tms0.y", "0.5"},
        {"tms0.w", "1"},
        {"tms0.h", "1"},
        {"tms0.iterations", "0"},
        {"tms0.vs", "0"},
        {"tms0.zoom", "-0.4"},
        {"tms0.stfx", "0.5"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "2"},
        {"tms0.zoom", "-0.7"},
        {"tms0.stfx", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Being able to build other types of dams increases the potential for complex behavior, which lets even beavers with very few emotions be cryptids."), 4);
    // more dam icons
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "5"},
        {"tms0.zoom", "-1.1"},
        {"tms0.stfx", "2"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "tms0", 0);
    cs.render_microblock();
    parse_tmstring(tmstrs[bigfoot], 3, 3, tm);
    // TODO: TNF fractal zoom to bigfoot
    cs.render_microblock();

    stage_macroblock(FileBlock("The first small cryptid ever discovered was Bigfoot, and it has 3 emotions and 3 different kinds of dams."), 3);
    as->reset();
    shown_tms = {bigfoot, probv_champ, wily_coyote, coyote_like};
    for (int i=0; i<4; i++) {
        pn = as->num_points;
        tn = as->text_anchors.size();
        as->add_point();
        as->add_point({pn});
        add_rect(as, pn, pn+1, labels[shown_tms[i]], i);
        parse_tmstring(tmstrs[shown_tms[i]], 3, 3, tm);
        tms[i]->set_tm(tm);
        tms[i]->default_everything(false, true);
        cs.manager.set({
            {"tms" + std::to_string(i) + ".x", std::to_string(0.3 + 0.2*(i&2))},
            {"tms" + std::to_string(i) + ".targetx", std::to_string(0.3 + 0.2*min(i,2) + 11.0/160*(i==3))},
            {"tms" + std::to_string(i) + ".y", std::to_string(0.8 - 0.6*(i&1))},
            {"tms" + std::to_string(i) + ".w", "0.2"},
            {"tms" + std::to_string(i) + ".h", "16 45 /"},
            {"tms" + std::to_string(i) + ".iterations", "75"},
            {"tms" + std::to_string(i) + ".vs", "0.4"},
            {"tms" + std::to_string(i) + ".zoom", "-3"},
            {"tms" + std::to_string(i) + ".stfx", "0"},
            {"tms" + std::to_string(i) + ".stfy", "25"},
        });
        as->manager.set({
            {"t" + std::to_string(tn) + "y", "<t" + std::to_string(tn) + "yabs> " + std::to_string(2*(i&1) - 1) + " *"},
        });
    }
    cs.move_to_front("tms0");
    cs.fade_subscene(MICRO, "tms0", 1);
    tms0->default_everything(true, true, true);
    cs.manager.set({
        {"tms0.x", "0.5"},
        {"tms0.y", "0.5"},
        {"tms0.w", "1.05"},
        {"tms0.h", "1.05"},
        {"tms0.iterations", "0"},
        {"tms0.vs", "0.2"},
        {"tms0.zoom", "-3.2"},
        {"tms0.stfy", "62"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "125"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.set({
        {"as.opacity", "1"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.x", "0.3"},
        {"tms0.y", "0.8"},
        {"tms0.w", "0.2"},
        {"tms0.h", "16 45 /"},
        {"tms0.vs", "0.4"},
        {"tms0.zoom", "-3"},
        {"tms0.stfy", "25"},
    });
    tms0->manager.transition(MICRO, {
        {"table_col_w", "0"},
        {"table_row_h", "0"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("It is one of only four unsolved beavers with 3 emotions and 3 types of dams!"), 2);
    for (int i=1; i<4; i++) cs.fade_subscene(MICRO, "tms" + std::to_string(i), 1);
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("The other 3 seem to also be cryptids, but they are on the less clear-cut end of the cryptid spectrum."), 3);
    // cryptid spectrum
    spect_start = as->num_points;
    create_spectrum(as, true);
    cs.render_microblock();
    for (int i=0; i<4; i++) cs.manager.transition(MICRO, "tms" + std::to_string(i) + ".x", "<tms" + std::to_string(i) + ".targetx>");
    cs.render_microblock();
    for (int i=0; i<4; i++) connect_to_spectrum(as, spect_start, (spect_start / 4) * i, spect_pos[shown_tms[i]], 0.5);
    cs.render_microblock();

    stage_macroblock(FileBlock("One of them is expected to eventually retire, but it's not an eclipse machine. Its strange behavior even led researchers to having a pretty tight estimate of when exactly it will retire - it should take close to 10^278 days."), 5);
    // expand probv champ to the whole screen
    cs.move_to_front("tms1");
    cs.manager.set({
        {"tms1.iterations", "150"},
    });
    cs.manager.transition(MICRO, {
        {"tms1.x", "0.5"},
        {"tms1.y", "0.5"},
        {"tms1.w", "1"},
        {"tms1.h", "1"},
        {"tms1.vs", "0.2"},
        {"tms1.zoom", "-3.2"},
        {"tms1.stfy", "62"},
    });
    for (int i=0; i<4; i++) cs.fade_subscene(MICRO, "tms" + std::to_string(i), i==1);
    cs.render_microblock();
    cs.manager.set({
        {"as.opacity", "0"},
    });
    as->reset();
    cs.render_microblock();
    // show cantor-set-like graphs from wiki?
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("The smallest cryptids with only 2 emotions have 5 dam types, and one of them is Hydra, the older sibling of Antihydra."), 4);
    // maybe do this on the cryptid spectrum after unexpanding probv 3x3 champ?
    cs.fade_subscene(MICRO, "tms1", 0);
    cs.render_microblock();
    parse_tmstring(tmstrs[hydra], 2, 5, tm);
    tms0->set_tm(tm);
    cs.fade_subscene(MICRO, "tms0", 1);
    cs.manager.set({
        {"tms0.x", "0.5"},
        {"tms0.y", "0.5"},
        {"tms0.w", "1"},
        {"tms0.h", "1"},
        {"tms0.iterations", "0"},
        {"tms0.vs", "0.2"},
        {"tms0.zoom", "-3.2"},
        {"tms0.stfx", "0"},
        {"tms0.stfy", "62"},
    });
    cs.manager.transition(MICRO, {
        {"tms0.iterations", "125"},
    });
    tms0->default_everything(true, true, true);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "tms0", 0);
    cs.render_microblock();

    stage_macroblock(FileBlock("Hydra computes almost the same thing as Antihydra, but the roles of odd and even numbers are swapped, and the starting numbers are different."), 3);
    // bring up the antihydra rules again but turn them into hydra rules
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_text(0, {"d\\text{ is even}:"});
    as->add_point();
    as->add_text(1, {"(t,d)\\ \\to\\ (t+2,\\lfloor\\frac{3}{2}d\\rfloor)"});
    as->add_point();
    as->add_text(2, {"d\\text{ is odd}:"});
    as->add_point();
    as->add_text(3, {"(t,d)\\ \\to\\ (t-1,\\lfloor\\frac{3}{2}d\\rfloor)"});
    as->add_point();
    as->add_text(4, {"t<0:"});
    as->add_point();
    as->add_icon(5);
    as->manager.set({
        {"transvar", "0.05"},
        {"p0x", "0.2 <transvar> +"},
        {"p0y", "0.25"},
        {"p1x", "0.55 <transvar> 4 * +"},
        {"p1y", "0.25"},
        {"p2x", "0.2 <transvar> +"},
        {"p2y", "0.5"},
        {"p3x", "0.55 <transvar> 4 * +"},
        {"p3y", "0.5"},
        {"p4x", "0.2 <transvar> +"},
        {"p4y", "0.75"},
        {"p5x", "0.55 <transvar> 4 * +"},
        {"p5y", "0.75"},
        {"t0h", "0.2"},
        {"t1h", "0.2"},
        {"t2h", "0.2"},
        {"t3h", "0.2"},
        {"t4h", "0.2"},
        {"i0h", "0.2"},
        {"i0id", std::to_string(icons_len-1)},
    });
    cs.render_microblock();
    as->manager.set({
        {"t1x", "0 pi {microblock_fraction} smoothlerp sin 0.04 *"},
        {"t3x", "0 pi {microblock_fraction} smoothlerp sin -0.04 *"},
    });
    as->manager.transition(MICRO, {
        {"p1y", "0.5"},
        {"p3y", "0.25"},
    });
    cs.render_microblock();
    as->manager.set({
        {"transvar", "0.05 0 {microblock_fraction} 2 * 1 min smoothlerp"},
        {"t1x", "0"},
        {"t3x", "0"},
    });
    pn = as->num_points;
    ln = as->line_endpoints.size();
    tn = as->text_anchors.size();
    as->add_point();
    as->add_point({pn});
    as->add_point({pn,pn+1});
    as->add_point({pn,pn+2});
    as->add_point({pn+1,pn+2});
    as->add_point({pn});
    as->add_point({pn+1});
    as->add_point({pn+2});
    as->add_point({pn,pn+2});
    as->add_point({pn+1,pn+2});
    as->add_line(pn,pn+5);
    as->add_line(pn+1,pn+6);
    as->add_line(pn+2,pn+7);
    as->add_line(pn+8,pn+9);
    antihydra_t = 0;
    antihydra_d = 3;
    for (int i=0; i<antihydra_table_n; i++) {
        as->add_text(pn+3, {i<antihydra_table_n-1 ? std::to_string(antihydra_t) : "..."});
        as->add_text(pn+4, {i<antihydra_table_n-1 ? std::to_string(antihydra_d) : "..."});
        as->add_line(pn,pn+1);
        as->manager.set({
            {"t" + std::to_string(tn+2*i  ) + "y", std::to_string((i+0.5)*antihydra_table_row_h)},
            {"t" + std::to_string(tn+2*i  ) + "o", "<tp> " + std::to_string(i) + " -"},
            {"t" + std::to_string(tn+2*i+1) + "y", std::to_string((i+0.5)*antihydra_table_row_h)},
            {"t" + std::to_string(tn+2*i+1) + "o", "<tp> " + std::to_string(i) + " -"},

            {"l" + std::to_string(ln+i+4) + "y", std::to_string(i*antihydra_table_row_h)},
            {"l" + std::to_string(ln+i+4) + "o", "<tp> " + std::to_string(i) + " >"},
        });
        antihydra_t += -1 + (antihydra_d & 1) * 3;
        antihydra_d += antihydra_d >> 1;
    }
    as->manager.set({
        {"tp", "-1"},
        {"p" + std::to_string(pn  ) + "x", "1 0.228125 -"},
        {"p" + std::to_string(pn  ) + "y", "0.05"},
        {"p" + std::to_string(pn+1) + "x", "0.2"},
        {"p" + std::to_string(pn+5) + "y", "<tp> " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+6) + "y", "<tp> " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+7) + "y", "<tp> " + std::to_string(antihydra_table_row_h) + " *"},
        {"p" + std::to_string(pn+8) + "w" + std::to_string(pn  ), "<tp> <tp> 1 - ceil -"},
        {"p" + std::to_string(pn+8) + "w" + std::to_string(pn+2), "<tp> ceil <tp> -"},
        {"p" + std::to_string(pn+9) + "w" + std::to_string(pn+1), "<tp> <tp> 1 - ceil -"},
        {"p" + std::to_string(pn+9) + "w" + std::to_string(pn+2), "<tp> ceil <tp> -"},
        {"l" + std::to_string(ln  ) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+1) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+2) + "o", "<tp> 0 >"},
        {"l" + std::to_string(ln+3) + "y", "<tp> ceil " + std::to_string(antihydra_table_row_h) + " *"},
    });
    for (int i=0; i<10; i++) as->manager.set("p" + std::to_string(pn+i) + "o", "0");
    as->manager.transition(MICRO, {
        {"tp", std::to_string(antihydra_table_n)},
        {"p1y", "0.5"},
        {"p3y", "0.25"},
    });
    cs.render_microblock();
    as->manager.set({
        {"transvar", "0"},
    });


    stage_macroblock(FileBlock("Now, all these small cryptids were discovered in the wild, as researchers were sifting through unsolved beavers. Someone decided to try to solve a specific beaver, and discovered a difficult math problem hidden within."), 5);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();
    cs.render_microblock();
    // latex "unsolved beaver --(analysis)-> difficult problem"
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_text(0, {"\\text{Unsolved beaver}"});
    as->add_point();
    as->add_text(1, {"\\text{Difficult problem}"});
    as->add_point();
    as->add_point();
    as->add_line(2, 3);
    as->add_point({2,3});
    as->add_text(4, {"\\text{analysis}"});
    as->add_text(4, {"\\text{programming}"});
    as->manager.set({
        {"p0x", "0.25"},
        {"p0y", "0.5"},
        {"p1x", "0.75"},
        {"p1y", "0.5"},
        {"p2x", "0.4"},
        {"p2y", "0.5"},
        {"p2o", "0"},
        {"p3x", "0.6"},
        {"p3y", "0.5"},
        {"p3o", "0"},
        {"t0h", "0.1"},
        {"t1o", "0"},
        {"t1h", "0.1"},
        {"t2y", "-0.05"},
        {"t2o", "0"},
        {"t2h", "0.06"},
        {"t3y", "0.05"},
        {"t3o", "0"},
        {"t3h", "0.06"},
        {"l0o", "0"},
        {"l0s", "1"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"t2o", "1"},
        {"l0o", "1"},
    });
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"t1o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("But before that, there already were some known cryptids. They were discovered by researchers constructing them."), 2);
    cs.render_microblock();
    as->manager.set({
        {"t0y", "0 pi {microblock_fraction} smoothlerp sin -0.17 *"},
        {"t1y", "0 pi {microblock_fraction} smoothlerp sin 0.17 *"},
    });
    as->manager.transition(MICRO, {
        {"p0x", "0.75"},
        {"p1x", "0.25"},
    });
    cs.render_microblock();
    as->manager.set({
        {"t0y", "0"},
        {"t1y", "0"},
    });

    stage_macroblock(FileBlock("Someone started with a difficult math problem, and worked backwards to find a beaver whose retirement depends on that problem. This is essentially programming."), 3);
    // latex "difficult problem --(programming)-> cryptid"
    as->manager.transition(MICRO, {
        {"t2o", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    as->manager.transition(MICRO, {
        {"t3o", "1"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(0.5), 1);
    cs.fade_subscene(MICRO, "as", 0);
    cs.render_microblock();
    as->reset();

    stage_macroblock(FileBlock("A few examples include the Riemann hypothesis, the consistency of ZFC set theory, the Goldbach conjecture, and an Erdos problem."), 5);
    // log-log plot
    cs.fade_subscene(MICRO, "as", 1);
    as->add_point();
    as->add_point({0});
    as->add_point({0});
    as->add_point({0});
    as->add_line(0, 2);
    as->add_line(0, 3);
    as->manager.set({
        {"p0x", "0.25"},
        {"p0y", "17 18 /"},
        {"p0o", "0"},
        {"p1x", "0.5"},
        {"p1y", "<p1x> -0.5625 /"},
        {"p1o", "0"},
        {"p2x", "<p1x>"},
        {"p2o", "0"},
        {"p3y", "<p1y>"},
        {"p3o", "0"},
	{"l" + std::to_string(as->line_endpoints.size()-2) + "s", "1"},
	{"l" + std::to_string(as->line_endpoints.size()-1) + "s", "1"},
        {"xaxistip", "1000"},
        {"yaxistip", "10"},
    });
    cs.render_microblock();
    // RH: 744x2,  Con(ZFC): 432x2,  goldbach: 26x2,  erdos: 5x4 & 15x2
    vector<int> constructed_num_states = {744, 432, 26, 15, 5};
    vector<int> constructed_num_symbols = {2, 2, 2, 2, 4};
    for (int i=0; i<5; i++) {
        as->add_point();
        as->manager.set({
            {"p" + std::to_string(as->num_points-1) + "xeq", "<p0xeq> <p1x> <xaxistip> log / <p" + std::to_string(as->num_points-1) + "x> log * +"},
            {"p" + std::to_string(as->num_points-1) + "yeq", "<p0yeq> <p1y> <yaxistip> log / <p" + std::to_string(as->num_points-1) + "y> log * +"},
            {"p" + std::to_string(as->num_points-1) + "x", std::to_string(constructed_num_states[i])},
            {"p" + std::to_string(as->num_points-1) + "y", std::to_string(constructed_num_symbols[i])},
            {"p" + std::to_string(as->num_points-1) + "o", "0"},
        });
        as->manager.transition(MICRO, {
            {"p" + std::to_string(as->num_points-1) + "o", "1"},
        });
        if (i != 3) cs.render_microblock();
    }

    stage_macroblock(FileBlock("The bias towards many emotions and few dam types is due to the similarity to other programming languages."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Even with only 2 dam types available, adding a line of code can often directly translate into adding a few more emotions. With a fixed number of emotions, additional dam types do not provide the same luxury."), 1);
    cs.render_microblock();

    // leave out?
    /*
    stage_macroblock(FileBlock("After Bigfoot and Hydra were discovered, people tried to construct equivalent cryptids with 2 symbols."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("For Hydra, they got one with 10 states, and for Bigfoot, they fit the problem in only 7 states. That's almost as small as 2-symbol cryptids get!"), 1);
    cs.render_microblock();
    */

    cs.remove_all_subscenes();
}

void outro(CompositeScene& cs) {
    /*
    Who will retire, and who will keep polishing their work forever? A simple question about beavers leading seemingly simple lives, but there's so much to learn about it, and so much to learn from it.
    This video is just the tip of the iceberg of all we know, and even that iceberg barely scratches the surface of the endless ocean of what there is to know.
    Of the beavers who do retire, which ones take the longest to do so? Which ones leave the most dams on the river? Who is the busiest beaver?
    */
}

void sponsor(CompositeScene& cs) {
    /*
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    */
}





void test(CompositeScene& cs) {
}



void render_video() {
    CompositeScene cs;
    icons_whnm = ivec4(512, 512, 6, 5);
    std::vector<std::string> pngnames = {"Wave.png", "Log.png", "Log2.png", "Log3.png", "Log4.png", "Ambitious.png", "Bitter.png", "Crying.png", "Disgusted.png", "Excited.png", "Fearful.png", "Left.png", "Right.png", "Sleep.png", "Retired.png"};
    icons = init_icons(pngnames, ivec2(icons_whnm.x, icons_whnm.y));
    icons_len = pngnames.size();

    //test(cs);
    preintro(cs);
    intro(cs);
    higher_domains(cs);
    cryptids(cs);
    outro(cs);
    sponsor(cs);

    cuda_free_pixels_on_device(icons);
}
