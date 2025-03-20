#include "../Scenes/Math/GraphScene.cpp"
#include "../DataObjects/HashableString.cpp"

void render_video() {
    //PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;

    Graph<HashableString> g;
    GraphScene gs(&g);

    gs.state_manager.set({
        {"q1", "<t> .1 * cos"},
        {"qi", "0"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"},
        {"surfaces_opacity", "0"},
        {"points_opacity", "0"},
        {"physics_multiplier", "3"},
    });
    g.add_node(make_shared<HashableString>("Life"));
    gs.inject_audio_and_render(SilenceSegment(1));
}
