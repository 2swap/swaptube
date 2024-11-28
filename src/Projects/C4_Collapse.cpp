#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void render_video() {
    PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;

    vector<string> variations{
        /*
    "41",
    "42",
    "4361",
    "4362",
    "4363",
    "4364",
    "4365",
    "4366",
    "4367",
    "4441",
    "4442",
    "4443",
    "444441",
    "444442",
    "444443",
    */
    "444444",
    };
    CompositeScene cs;
    cs.inject_audio(AudioSegment(variations.size() * 2), variations.size());
    for(const string& variation : variations){
        Graph<C4Board> g;
        C4GraphScene gs(&g, variation, TRIM_STEADY_STATES);
        LatexScene ls_opening(latex_text("Opening: "+variation), 1, .2, .1);
        LatexScene ls_size(latex_text("Node count: "+to_string(g.size())), 1, .2, .1);
        C4Scene c4s(variation, .2, .4);
        cs.add_scene(&gs, "gs");
        cs.add_scene(&ls_opening, "ls_opening", .1, .05);
        cs.add_scene(&ls_size, "ls_size", .1, .12);
        cs.add_scene(&c4s, "c4s", .1, .26);
        //ValidateC4Graph(g);

        g.dimensions = 3;

        StateSet state{
            {"q1", "<t> .1 * cos"},
            {"qi", "0"},
            {"qj", "<t> .1 * sin"},
            {"qk", "0"},
            {"surfaces_opacity", "0"},
            {"points_opacity", "0"},
            {"physics_multiplier", "40"},
        };
        gs.state_manager.set(state);

        cs.render();
        g.render_json("c4_" + variation + ".json");
        cs.remove_all_scenes();
    }
}
