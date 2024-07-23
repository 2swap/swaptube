#include "../io/PathManager.cpp"

PathManager("C4_Manual_Tree");

#include "../scenes/Connect4/c4_graph_scene.cpp"

void render_video() {
    FOR_REAL = true; // Whether we should actually be writing any AV output
    PRINT_TO_TERMINAL = true;
    const int WIDTH_BASE = 640;
    const int HEIGHT_BASE = 360;
    const int MULT = 1;
    const int VIDEO_WIDTH = WIDTH_BASE*MULT;
    const int VIDEO_HEIGHT = HEIGHT_BASE*MULT;
    const int VIDEO_FRAMERATE = 30;

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
