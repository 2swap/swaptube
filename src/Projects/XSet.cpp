using namespace std;
#include <string>
const string project_name = "XSet";
const int width_base = 640;
const int height_base = 360;
const float mult = 2;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

//#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

void intro() {
    MandelbrotScene ms;
    ms.state_manager.set(unordered_map<string,string>{
        {"zoom_r", "2"},
        {"zoom_i", "0"},
        {"max_iterations", "100"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"side_panel", "0"},
    });
    ms.inject_audio_and_render(AudioSegment(1));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"side_panel", "1"},
    });
    ms.inject_audio_and_render(AudioSegment(1));
    ms.inject_audio_and_render(AudioSegment(2));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "3"},
    });
    ms.inject_audio_and_render(AudioSegment(1));
    ms.inject_audio_and_render(AudioSegment(2));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_z_r", ".5"},
        {"seed_z_i", "-.3"},
    });
    ms.inject_audio_and_render(AudioSegment(1));
    ms.inject_audio_and_render(AudioSegment(2));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", ".2"},
        {"seed_c_i", "-.6"},
    });
    ms.inject_audio_and_render(AudioSegment(1));
    ms.inject_audio_and_render(AudioSegment(2));
}

int main() {
    Timer timer;
    FOR_REAL = true;
    PRINT_TO_TERMINAL = true;
    intro();
    return 0;
}

