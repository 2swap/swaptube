#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
//#include "../DataObjects/Connect4/TreeValidator.cpp"

void render_video() {
    try{
        //PRINT_TO_TERMINAL = false;
        SAVE_FRAME_PNGS = false;

        CompositeScene cs;

        Graph g;
        string variation = "";
        shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(&g, false, variation, TRIM_STEADY_STATES);
        shared_ptr<LatexScene> ls_opening = make_shared<LatexScene>(latex_text("Opening: "+variation), 1, .2, .1);
        shared_ptr<LatexScene> ls_size = make_shared<LatexScene>(latex_text("Node count: "+to_string(g.size())), 1, .2, .1);
        shared_ptr<C4Scene> c4s = make_shared<C4Scene>(variation, .2, .4);

        StateSet default_graph_state{
            {"q1", "1"},
            {"qi", "<t> .2 * cos"},
            {"qj", "<t> .314 * sin"},
            {"qk", "0"}, // Camera orientation quaternion
            {"decay",".98"},
            {"dimensions","3.98"},
            {"surfaces_opacity","0"},
            {"points_opacity","0"},
            {"physics_multiplier","100"}, // How many times to iterate the graph-spreader
        };
        gs->state_manager.set(default_graph_state);

        cs.add_scene(gs, "gs");
        cs.add_scene(ls_opening, "ls_opening", .1, .05);
        cs.add_scene(ls_size, "ls_size", .1, .12);
        cs.add_scene(c4s, "c4s", .1, .26);
        //ValidateC4Graph(g);

        cs.stage_macroblock(SilenceBlock(20), 3);
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"dimensions", "3"}});
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"decay", ".5"}});
        cs.render_microblock();
        cs.render_microblock();
        //g.render_json("../../Klotski/viewer/data/c4_"+variation+".json");

    } catch (exception& e){
        movecache.WriteCache();
        fhourstonesCache.WriteCache();
    }
}
