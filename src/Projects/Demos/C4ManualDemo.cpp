#include "../Scenes/Connect4/C4GraphScene.cpp"

void render_video() {
    Graph g;
    C4GraphScene gs(&g, false, "444", MANUAL);

    gs.manager.set({
        {"q1", "{t} 4 / cos"},
        {"qi", "0"},
        {"qj", "{t} -4 / sin"},
        {"qk", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "1"},
        {"physics_multiplier", "1"},
        {"d", "2"},
        {"dimensions", "2"},
    });
    gs.manager.transition(MICRO, {
        {"q1", "{t} 4 / cos"},
        {"qi", "0"},
        {"qj", "{t} -4 / sin"},
        {"qk", "0"},
        {"d", "8"},
    });
    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board("444" + to_string(i)));
        g.add_missing_edges();
    }
    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();
}
