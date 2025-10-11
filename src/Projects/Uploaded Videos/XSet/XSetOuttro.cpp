using namespace std;
#include <string>
const string project_name = "XSetOuttro";
const int width_base = 640;
const int height_base = 360;
const float mult = 6;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"

void intro() {
    CompositeScene cs;
    MandelbrotScene ms;
    cs.add_scene(&ms, "ms");
    ms.state_manager.set(unordered_map<string,string>{
        {"max_iterations", "150 2 <zoom_exp> -3 / ^ *"},
        {"gradation", "1"},
        {"side_panel", "0"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
        {"zoom_r", "2 <zoom_exp> ^"},
        {"zoom_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"seed_z_r", "3.55"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
        {"zoom_exp", "1"},
        {"internal_shade", "1"},
        {"breath", "{t} .73 + 3 / sin 2 / "},
    });
    TwoswapScene tss;
    tss.state_manager.set(unordered_map<string,string>{
        {"circle_opacity", "0"},
    });
    cs.add_scene(&tss, "tss");
    cs.state_manager.set(unordered_map<string,string>{
        {"tss.opacity", "0"},
    });
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "{t} sin .75 *"},
        {"seed_c_i", "{t} cos .75 *"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"tss.opacity", "1"},
    });
    cs.stage_macroblock(AudioSegment("This has been 2swap."), 3);
    cs.render_microblock();
    cs.render_microblock();
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
    });
    cs.render_microblock();
    PngScene note("note", 0.18, 0.18);
    LatexScene seef(latex_text("6884"), 1, .6, .27);
    cs.add_scene_fade_in(&seef, "seef", 0.6, 0.73);
    cs.add_scene_fade_in(&note, "note", 0.44, 0.73);
    cs.stage_macroblock_and_render(AudioSegment("with music by 6884"));
    cs.stage_macroblock_and_render(AudioSegment(1));
}

int main() {
    Timer timer;
    PRINT_TO_TERMINAL = true;
    signal(SIGINT, signal_handler);
    try {
        intro();
    }
    catch(std::exception& e) {
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
    }
    return 0;
}

