#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/LambdaScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"

void render_video() {
    Graph<C4Board> g;
    g.dimensions = 3;
    C4GraphScene gs(&g, "43667555535331111113322", SIMPLE_WEAK);

    gs.state_manager.set(unordered_map<string, string>{
        {"q1", "<t> .1 * cos"},
        {"qi", "0"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"},
        {"d", "150"},
        {"surfaces_opacity", "0"},
        {"points_opacity", "0"},
        {"physics_multiplier", "1"},
    });
    gs.stage_macroblock_and_render(AudioSegment(4));
}

