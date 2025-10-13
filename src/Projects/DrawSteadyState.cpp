#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void render_video() {
    vector<string> variations = {
        "452333534244",
        "444442222341",
        "43637565665332421344",
        "452122414442422111",
        "4444473226622667",
        "45233212263442"
    };
    Graph g;
    C4GraphScene gs(&g, false, "436675535335733556", NAIVE_WEAK);

    gs.state.set({
        {"q1", "1"},
        {"qi", "{t} 3 / sin"},
        {"qj", "{t} 2 / cos"},
        {"qk", "0"},
        {"decay",".6"},
        {"dimensions","3"},
        {"surfaces_opacity","0"},
        {"points_opacity","0"},
        {"physics_multiplier","100"},
    });

    // Iterate over all graph nodes and search for steadystates
    int count = 0;

    gs.stage_macroblock(SilenceBlock(6), 1);
    gs.render_microblock();

    cout << g.size() << " nodes" << endl;
}
