#include "../Scenes/Connect4/Connect4GraphScene.cpp"

void render_video() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.5;
    g.gravity_strength = 0;
    g.dimensions = 2;
    C4GraphScene gs(&g, "444", MANUAL);

    gs.state_manager.set(StateSet{
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
        {"physics_multiplier", "1"},
        {"d", "2"},
    });
    gs.state_manager.microblock_transition(StateSet{
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"},
        {"d", "8"},
    });
    gs.inject_audio_and_render(AudioSegment(1));
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board("444" + to_string(i)));
    }
    g.dimensions = 3;
    gs.inject_audio_and_render(AudioSegment(1));
}
