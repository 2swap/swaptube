#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"

void render_video() {
    PRINT_TO_TERMINAL = false;
    Graph<C4Board> g;
    g.dimensions = 3;
    string variation = "44444453";
    C4GraphScene gs(&g, variation, TRIM_STEADY_STATES);

    gs.state_manager.set(StateSet{
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
    //g.render_json(variation + ".json");
}

