using namespace std;
#include <string>
const string project_name = "XSetThumbnail";
const int width_base = 640;
const int height_base = 360;
const float mult = 3;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void intro() {
    MandelbrotScene ms;
    ms.state.set(unordered_map<string,string>{
        {"zoom_r", "2 <zoom_exp> ^"},
        {"zoom_i", "0"},
        {"max_iterations", "{t} 2 + sin 5 * 15 +"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"side_panel", "0"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
        {"internal_shade", "0"},
        {"gradation", "0"},
        {"seed_z_r", "0.5"},
        {"seed_z_i", "0.5"},
        {"seed_c_r", "0.08"},
        {"seed_c_i", "0.03"},
        {"zoom_exp", "-4.5"},
        {"breath", "0"},
        {"seed_x_r", "0"},
        {"seed_x_i", "{t} 2 + cos .5 * 3 +"},
    });
    ms.state.set(unordered_map<string,string>{
        {"zoom_r", "2 <zoom_exp> ^"},
        {"zoom_i", "0"},
        {"max_iterations", "12"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"side_panel", "0"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
        {"internal_shade", "0"},
        {"gradation", "0"},
        {"seed_z_r", "0.33333"},
        {"seed_z_i", "0"},
        {"seed_c_r", "0.13"},
        {"seed_c_i", "0.045"},
        {"zoom_exp", "-4.8"},
        {"breath", "0"},
        {"seed_x_r", "0"},
        {"seed_x_i", "3"},
    });
    ms.stage_macroblock_and_render(AudioSegment(2));
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

