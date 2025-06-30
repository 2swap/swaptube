#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
void render_video() {
    CompositeScene cs;
        Graph fg;
        shared_ptr<C4GraphScene> fgs = make_shared<C4GraphScene>(&fg, false, "", TRIM_STEADY_STATES, .5, 1);

        StateSet c4_default_graph_state{
            {"q1", "1"},
            {"qi", "<t> .2 * cos"},
            {"qj", "<t> .314 * sin"},
            {"qk", "0"}, // Camera orientation quaternion
            {"decay",".6"},
            {"dimensions","3.98"},
            {"surfaces_opacity","0"},
            {"points_opacity","0"},
            {"physics_multiplier","100"}, // How many times to iterate the graph-spreader
        };
        fgs->state_manager.set(c4_default_graph_state);
        cs.add_scene_fade_in(MICRO, fgs, "fgs", .75, .5);
    cs.stage_macroblock(FileBlock("Stay tuned to see how strategy board game solutions take that same form."), 2);
    cs.render_microblock();
    cs.render_microblock();
}
