#include "../Scenes/Math/Beavers/BeaverGridTNFScene.h"
#include "../Scenes/Math/Beavers/BeaverGridTNF3DScene.h"
#include "../Scenes/Math/Beavers/BeaverTNF3DScene.h"
#include "../Scenes/Math/Beavers/BeaverIndividualScene.h"
#include "../Core/State/BezierStateCurve.h"
#include "../IO/PNG.h"
#include <vector>
#include <string>

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
    bs.manager.set("highlight_x", std::to_string(center.x));
    bs.manager.set("highlight_y", std::to_string(center.y));
    bs.manager.set("highlight_z", std::to_string(center.z));
    bs.manager.set("highlight_intensity", "0.3");

    vector<StateSet> waypoints =
{{{"target_x","0.632116622776"}, {"target_y","-1.85697329369"}, {"target_z","0.467810611747"}, {"camera_distance","0"}, {"q1","0.546068887831"}, {"qi","-0.467719337537"}, {"qj","0.503367147438"}, {"qk","0.479237838571"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.716985108098"}, {"target_y","-1.31377495136"}, {"target_z","0.499280958408"}, {"camera_distance","0"}, {"q1","0.531793699777"}, {"qi","-0.483888489055"}, {"qj","0.488765670525"}, {"qk","0.49412094709"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.736142328592"}, {"target_y","-0.854843801936"}, {"target_z","0.524084203926"}, {"camera_distance","0"}, {"q1","0.541658475285"}, {"qi","-0.503473853901"}, {"qj","0.468909148533"}, {"qk","0.482953812505"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.744116150534"}, {"target_y","-0.487452958922"}, {"target_z","0.543689984527"}, {"camera_distance","0"}, {"q1","0.569643628032"}, {"qi","-0.530685544028"}, {"qj","0.437875055808"}, {"qk","0.449604744084"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.754083427961"}, {"target_y","-0.0282144051539"}, {"target_z","0.568197210279"}, {"camera_distance","0"}, {"q1","0.666082528981"}, {"qi","-0.625418352296"}, {"qj","0.286729133068"}, {"qk","0.288049220528"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.760063794418"}, {"target_y","0.247328727107"}, {"target_z","0.58290154573"}, {"camera_distance","0"}, {"q1","0.667893215965"}, {"qi","-0.638899207447"}, {"qj","-0.255283203252"}, {"qk","-0.283825546646"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.766044160874"}, {"target_y","0.522871859367"}, {"target_z","0.597605881181"}, {"camera_distance","0"}, {"q1","0.00650401904364"}, {"qi","-0.0210932744817"}, {"qj","-0.687689453831"}, {"qk","-0.725669336955"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.77202452733"}, {"target_y","0.798414991628"}, {"target_z","0.612310216632"}, {"camera_distance","0"}, {"q1","-0.382696989357"}, {"qi","0.500746260569"}, {"qj","-0.598824194591"}, {"qk","-0.494171813073"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.754416431538"}, {"target_y","0.957778855941"}, {"target_z","0.522037775025"}, {"camera_distance","0"}, {"q1","-0.181153335611"}, {"qi","0.866091449425"}, {"qj","-0.431102403297"}, {"qk","-0.176691222477"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.747734784607"}, {"target_y","1.01020297201"}, {"target_z","0.446731283613"}, {"camera_distance","0"}, {"q1","0.218180023907"}, {"qi","0.943673966349"}, {"qj","-0.11734204073"}, {"qk","-0.219334830521"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.820203221681"}, {"target_y","1.03063325829"}, {"target_z","0.251070521574"}, {"camera_distance","0"}, {"q1","0.771589766504"}, {"qi","0.619902219385"}, {"qj","-0.137421233883"}, {"qk","0.0385470505421"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.8603822028"}, {"target_y","1.04252723217"}, {"target_z","0.0925329793921"}, {"camera_distance","0"}, {"q1","0.916690686332"}, {"qi","0.342916023433"}, {"qj","-0.194522026609"}, {"qk","-0.0651764346129"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.91037235078"}, {"target_y","0.898516030643"}, {"target_z","-0.00994174813817"}, {"camera_distance","0"}, {"q1","0.984458985025"}, {"qi","0.164639923048"}, {"qj","-0.0515647462721"}, {"qk","0.0327914544289"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.93012642519"}, {"target_y","0.706977001777"}, {"target_z","-0.136130286846"}, {"camera_distance","0"}, {"q1","0.994758650222"}, {"qi","0.026600130418"}, {"qj","-0.0955055059305"}, {"qk","0.0250271693959"}, {"fov","0.39"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.532225384591"}, {"target_z","-0.512334217251"}, {"camera_distance","0"}, {"q1","0.992713911655"}, {"qi","-0.109641106547"}, {"qj","-0.0494179570741"}, {"qk","0.00746879371839"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.528225651252"}, {"target_z","-0.312374215918"}, {"camera_distance","0"}, {"q1","0.999200106661"}, {"qi","0.0399893341866"}, {"qj","4.06514831055e-22"}, {"qk","1.35464289763e-20"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.524225917913"}, {"target_z","-0.112414214585"}, {"camera_distance","0"}, {"q1","0.999950000417"}, {"qi","0.00999983333417"}, {"qj","-1.03443502872e-36"}, {"qk","1.35525271561e-20"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.520226184575"}, {"target_z","0.0875457867486"}, {"camera_distance","0"}, {"q1","0.999950000417"}, {"qi","0.00999983333417"}, {"qj","-1.03443502872e-36"}, {"qk","1.35525271561e-20"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.0508790228581"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.516565669745"}, {"target_z","0.270547124136"}, {"camera_distance","0"}, {"q1","0.999950000417"}, {"qi","0.00999983333417"}, {"qj","-1.03443502872e-36"}, {"qk","1.35525271561e-20"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.0144002595138"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.515058492862"}, {"target_z","0.3458959202"}, {"camera_distance","0"}, {"q1","0.999950000417"}, {"qi","0.00999983333417"}, {"qj","-1.03443502872e-36"}, {"qk","1.35525271561e-20"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.0263832738939"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.899521255663"}, {"target_y","0.513829080423"}, {"target_z","0.436987221144"}, {"camera_distance","0"}, {"q1","0.987227283376"}, {"qi","0.159318206614"}, {"qj","2.02526434850e-21"}, {"qk","1.34003468849e-20"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.901424689488"}, {"target_y","0.494760127278"}, {"target_z","0.4686730404"}, {"camera_distance","0"}, {"q1","0.918624449663"}, {"qi","0.320848625229"}, {"qj","-0.0526386139276"}, {"qk","0.224531638068"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.907061691527"}, {"target_y","0.467419768225"}, {"target_z","0.501148674307"}, {"camera_distance","0"}, {"q1","0.738273500371"}, {"qi","0.391692654691"}, {"qj","-0.0867257687472"}, {"qk","0.542224809416"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.912011255273"}, {"target_y","0.463901184516"}, {"target_z","0.503745344154"}, {"camera_distance","0"}, {"q1","0.12322153832"}, {"qi","0.500598943277"}, {"qj","-0.211343718714"}, {"qk","0.830392065861"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.0124530992566"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.937981672072"}, {"target_y","0.448092860022"}, {"target_z","0.517390983713"}, {"camera_distance","0"}, {"q1","-0.234793454579"}, {"qi","0.497732045132"}, {"qj","-0.54197996474"}, {"qk","0.635131925474"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.0183059373346"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.963860264126"}, {"target_y","0.434050901399"}, {"target_z","0.513259076565"}, {"camera_distance","0"}, {"q1","-0.0727498831756"}, {"qi","0.378249838894"}, {"qj","-0.902964584422"}, {"qk","0.190497961024"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.00815027562607"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.979008794112"}, {"target_y","0.430100567771"}, {"target_z","0.493671645514"}, {"camera_distance","0"}, {"q1","-0.0585019390631"}, {"qi","-0.00442495652377"}, {"qj","-0.998046544535"}, {"qk","0.0214717914503"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.0138936074524"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.981305402524"}, {"target_y","0.428052641349"}, {"target_z","0.440679510271"}, {"camera_distance","0"}, {"q1","-0.0186287115174"}, {"qi","-0.00410904291744"}, {"qj","-0.999817163674"}, {"qk","0.00131380966235"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.109821223201"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.981461343316"}, {"target_y","0.42802040312"}, {"target_z","0.428170410016"}, {"camera_distance","0"}, {"q1","0.00137002431127"}, {"qi","-0.00413449557778"}, {"qj","-0.999989756299"}, {"qk","0.00123137152944"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.123406015453"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.981444178314"}, {"target_y","0.428005103121"}, {"target_z","0.421929093635"}, {"camera_distance","0"}, {"q1","0.00137002431127"}, {"qi","-0.00413449557778"}, {"qj","-0.999989756299"}, {"qk","0.00123137152944"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.139456154984"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.981404188749"}, {"target_y","0.427969458482"}, {"target_z","0.407388604364"}, {"camera_distance","0"}, {"q1","0.00137002431127"}, {"qi","-0.00413449557778"}, {"qj","-0.999989756299"}, {"qk","0.00123137152944"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.236941060287"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},
 {{"target_x","0.981401547279"}, {"target_y","0.427967104011"}, {"target_z","0.406428146913"}, {"camera_distance","0"}, {"q1","0.00137002431127"}, {"qi","-0.00413449557778"}, {"qj","-0.999989756299"}, {"qk","0.00123137152944"}, {"fov","1"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","-0.32647088582"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}}}
    ;
/*
{{{"target_x","0.632116622776"}, {"target_y","-1.85697329369"}, {"target_z","0.467810611747"}, {"camera_distance","0"}, {"q1","0.546068887831"}, {"qi","-0.467719337537"}, {"qj","0.503367147438"}, {"qk","0.479237838571"}, {"fov","1
"}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                          
 {{"target_x","0.705490775801"}, {"target_y","-1.58913364102"}, {"target_z","0.484399011098"}, {"camera_distance","0"}, {"q1","0.531793699777"}, {"qi","-0.483888489055"}, {"qj","0.488765670525"}, {"qk","0.49412094709"}, {"fov","1"}, {"zoo
m","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                   
 {{"target_x","0.716985108098"}, {"target_y","-1.31377495136"}, {"target_z","0.499280958408"}, {"camera_distance","0"}, {"q1","0.531793699777"}, {"qi","-0.483888489055"}, {"qj","0.488765670525"}, {"qk","0.49412094709"}, {"fov","1"}, {"zoo
m","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                   
 {{"target_x","0.736142328592"}, {"target_y","-0.854843801936"}, {"target_z","0.524084203926"}, {"camera_distance","0"}, {"q1","0.541658475285"}, {"qi","-0.503473853901"}, {"qj","0.468909148533"}, {"qk","0.482953812505"}, {"fov","1"}, {"z
oom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                 
 {{"target_x","0.744116150534"}, {"target_y","-0.487452958922"}, {"target_z","0.543689984527"}, {"camera_distance","0"}, {"q1","0.569643628032"}, {"qi","-0.530685544028"}, {"qj","0.437875055808"}, {"qk","0.449604744084"}, {"fov","1"}, {"z
oom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                 
 {{"target_x","0.754083427961"}, {"target_y","-0.0282144051539"}, {"target_z","0.568197210279"}, {"camera_distance","0"}, {"q1","0.666082528981"}, {"qi","-0.625418352296"}, {"qj","0.286729133068"}, {"qk","0.288049220528"}, {"fov","1"}, {"
zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                 {{"target_x","0.760063794418"}, {"target_y","0.247328727107"}, {"target_z","0.58290154573"}, {"camera_distance","0"}, {"q1","0.667893215965"}, {"qi","-0.638899207447"}, {"qj","-0.255283203252"}, {"qk","-0.283825546646"}, {"fov","1"}, {"z
oom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                  {{"target_x","0.766044160874"}, {"target_y","0.522871859367"}, {"target_z","0.597605881181"}, {"camera_distance","0"}, {"q1","0.00650401904364"}, {"qi","-0.0210932744817"}, {"qj","-0.687689453831"}, {"qk","-0.725669336955"}, {"fov","1"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                              {{"target_x","0.77202452733"}, {"target_y","0.798414991628"}, {"target_z","0.612310216632"}, {"camera_distance","0"}, {"q1","-0.382696989357"}, {"qi","0.500746260569"}, {"qj","-0.598824194591"}, {"qk","-0.494171813073"}, {"fov","0.39"}, 
{"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                               {{"target_x","0.754416431538"}, {"target_y","0.957778855941"}, {"target_z","0.522037775025"}, {"camera_distance","0"}, {"q1","-0.181153335611"}, {"qi","0.866091449425"}, {"qj","-0.431102403297"}, {"qk","-0.176691222477"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                              {{"target_x","0.747734784607"}, {"target_y","1.01020297201"}, {"target_z","0.446731283613"}, {"camera_distance","0"}, {"q1","0.218180023907"}, {"qi","0.943673966349"}, {"qj","-0.11734204073"}, {"qk","-0.219334830521"}, {"fov","0.39"}, {"
zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                                 {{"target_x","0.731454025967"}, {"target_y","1.05896183647"}, {"target_z","0.210215562389"}, {"camera_distance","0"}, {"q1","0.882274898888"}, {"qi","0.469285900675"}, {"qj","-0.0351738341589"}, {"qk","-0.0111600900778"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                              {{"target_x","0.733407782251"}, {"target_y","1.01775984809"}, {"target_z","0.130357056004"}, {"camera_distance","0"}, {"q1","0.945771639705"}, {"qi","0.322729390218"}, {"qj","-0.0329465632058"}, {"qk","-0.0166213775877"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                              {{"target_x","0.689023546058"}, {"target_y","1.01995995788"}, {"target_z","-0.0103056786529"}, {"camera_distance","0"}, {"q1","0.969743620039"}, {"qi","0.223674832191"}, {"qj","0.0888354201873"}, {"qk","0.0409285836372"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","50"}, {"max_tnf_depth","15"}},                                                              {{"target_x","0.749213962083"}, {"target_y","0.77305422931"}, {"target_z","-0.154988371041"}, {"camera_distance","0"}, {"q1","0.991061175297"}, {"qi","-0.0186680153465"}, {"qj","-0.119858092126"}, {"qk","-0.055527378592"}, {"fov","0.39"}
, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                             {{"target_x","0.753075120952"}, {"target_y","0.750757773985"}, {"target_z","0.0630574574493"}, {"camera_distance","0"}, {"q1","0.99454303143"}, {"qi","-0.0415037990332"}, {"qj","-0.0493486800188"}, {"qk","0.0820140297783"}, {"fov","0.39"
}, {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                            {{"target_x","0.743628130618"}, {"target_y","0.757459179384"}, {"target_z","0.15230904309"}, {"camera_distance","0"}, {"q1","0.901898234668"}, {"qi","-0.0562275421024"}, {"qj","-0.0315645538998"}, {"qk","0.427108553823"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                              {{"target_x","0.769510709944"}, {"target_y","0.734309603718"}, {"target_z","0.324009681403"}, {"camera_distance","0"}, {"q1","0.712657405161"}, {"qi","0.164169050876"}, {"qj","0.0157976735763"}, {"qk","0.681849234885"}, {"fov","0.39"}, {
"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                                {{"target_x","0.775810113375"}, {"target_y","0.720747574079"}, {"target_z","0.387635687212"}, {"camera_distance","0"}, {"q1","0.728593423097"}, {"qi","0.0631163862697"}, {"qj","-0.0795042568754"}, {"qk","0.677382475963"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                              {{"target_x","0.773766317893"}, {"target_y","0.725968606267"}, {"target_z","0.541793281002"}, {"camera_distance","0"}, {"q1","0.693827857613"}, {"qi","0.0590639505099"}, {"qj","-0.0825594018099"}, {"qk","0.712950418278"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                              {{"target_x","0.770309110983"}, {"target_y","0.724431527969"}, {"target_z","0.639392897347"}, {"camera_distance","0"}, {"q1","0.693827857613"}, {"qi","0.0590639505099"}, {"qj","-0.0825594018099"}, {"qk","0.712950418278"}, {"fov","0.39"},
 {"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                              {{"target_x","0.773663996911"}, {"target_y","0.708531290596"}, {"target_z","0.678121790849"}, {"camera_distance","0"}, {"q1","0.320749235296"}, {"qi","0.369584175352"}, {"qj","-0.518325969872"}, {"qk","0.70133134419"}, {"fov","0.39"}, {"
zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                                 {{"target_x","0.773509849906"}, {"target_y","0.692163088991"}, {"target_z","0.686761592712"}, {"camera_distance","0"}, {"q1","0.155201288123"}, {"qi","0.269060306917"}, {"qj","-0.585572344567"}, {"qk","0.748748382759"}, {"fov","0.39"}, {
"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}},                                                                {{"target_x","0.772344613559"}, {"target_y","0.674222667159"}, {"target_z","0.692826532013"}, {"camera_distance","0"}, {"q1","0.0872767287163"}, {"qi","0.215340955239"}, {"qj","-0.607385126981"}, {"qk","0.759667264757"}, {"fov","0.39"}, 
{"zoom","0"}, {"scale_x","1"}, {"scale_y","1"}, {"scale_z","1"}, {"brightness_offset","0.13815510558"}, {"color_source_depth","3"}, {"max_steps","38"}, {"max_tnf_depth","10"}}}
*/

    BezierStateCurve bsc(waypoints);
    stage_macroblock(SilenceBlock(waypoints.size()), waypoints.size()-1);
    while(remaining_microblocks_in_macroblock) {
        bs.manager.set(bsc.pop_next_state_set());
        bs.render_microblock();
    }
}

void render_video() {
    TNF3Dtest();
}
