using namespace std;
#include <string>
const string project_name = "C4_Manual_Tree";
#include "../io/PathManager.cpp"



const int width_base = 640;
const int height_base = 360;
const int mult = 1;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"

#include "../scenes/Connect4/c4_graph_scene.cpp"
#include "../misc/Timer.cpp"

void render_video() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.5;
    g.gravity_strength = 0;
    g.dimensions = 2;
    g.sqrty = true;
    C4GraphScene gs(&g, "444", MANUAL, VIDEO_WIDTH, VIDEO_HEIGHT);
    gs.physics_multiplier = 1;

    gs.dag.add_equations(std::unordered_map<std::string, std::string>{
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "1"},
        {"d", "2"}
    });
    gs.dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"},
        {"d", "8"}
    });
    gs.inject_audio_and_render(AudioSegment(1));
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board("444" + to_string(i)));
    }
    g.dimensions = 3;
    gs.inject_audio_and_render(AudioSegment(1));
}
int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}
