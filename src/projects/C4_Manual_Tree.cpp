#include "../scenes/Connect4/c4_graph_scene.cpp"

void render_video() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.5;
    g.gravity_strength = 0;
    g.dimensions = 2;
    g.sqrty = true;
    C4GraphScene gs(&g, "444", MANUAL);
    gs.physics_multiplier = 1;

    dag.add_equations(std::unordered_map<std::string, std::string>{
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
    dag.add_transitions(std::unordered_map<std::string, std::string>{
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


    cout << g.size() << endl;
}
