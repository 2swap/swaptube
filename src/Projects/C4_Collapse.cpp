#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"

void render_video() {
    PRINT_TO_TERMINAL = false;

    Graph<C4Board> g;
    string variation = "44444435";
    C4GraphScene gs(&g, variation, TRIM_STEADY_STATES);

    Graph<C4Board> g2;
    string variation2 = "44444453";
    C4GraphScene gs2(&g2, variation2, TRIM_STEADY_STATES);

    g2.dimensions = g.dimensions = 3;

    StateSet state = StateSet{
        {"q1", "<t> .1 * cos"},
        {"qi", "0"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"},
        {"d", "600"},
        {"surfaces_opacity", "0"},
        {"points_opacity", "0"},
        {"physics_multiplier", "10"},
    };
    gs.state_manager.set(state);
    gs2.state_manager.set(state);

    gs.inject_audio_and_render(AudioSegment(2));
    gs2.inject_audio_and_render(AudioSegment(2));
    cout << g.size() << " <- SIZE COMPARISON -> " << g2.size() << endl;
    g.render_json(variation + ".json");
    g2.render_json(variation2 + ".json");
}

