using namespace std;
#include <string>
const string project_name = "C4_Collapse";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const int mult = 2;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/LambdaScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"

void render_video() {
    Graph<C4Board> g;
    g.dimensions = 3;
    string variation = "4364";
    C4GraphScene gs(&g, variation, TRIM_STEADY_STATES);

    gs.state_manager.set(unordered_map<string, string>{
        {"q1", "<t> .1 * cos"},
        {"qi", "0"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"},
        {"d", "600"},
        {"surfaces_opacity", "0"},
        {"points_opacity", "0"},
        {"physics_multiplier", "1"},
    });
    gs.inject_audio_and_render(AudioSegment(5));
    cout << "GRAPH SIZE: " << g.size() << endl;
    g.render_json(variation + ".json");
}

int main() {
    Timer timer;
    steady_state_unit_tests();
    render_video();
    return 0;
}

