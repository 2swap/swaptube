#include "../../Klotski/C4Board.cpp"
void render_video() {
    GraphScene<C4Board> gs(&graph);

    c4_branch_mode = UNION_WEAK;
    std::string nodestr = "4366755553533111111332222666675";
    graph.add_to_stack(new C4Board(nodestr));
    graph.expand_graph_dfs();
    graph.make_edges_bidirectional();
    gs.graph_to_3d();

    VariableScene v(&gs);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"x", "t sin 150 *"},
        {"y", "0"},
        {"z", "t cos 150 *"},
        {"q1", "t 4 / cos"},
        {"q2", "0"},
        {"q3", "t -4 / sin"},
        {"q4", "0"}
    });
    v.inject_audio_and_render(AudioSegment(10));
}