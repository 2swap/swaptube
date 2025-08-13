#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void render_video() {
    try{
        SAVE_FRAME_PNGS = false;

        CompositeScene cs;

        Graph g;
        string variation = "";
        shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(&g, false, variation, TRIM_STEADY_STATES);

        gs->state_manager.set({
            {"q1", "1"},
            {"qi", "<t> .2 * cos"},
            {"qj", "<t> .314 * sin"},
            {"qk", "0"},
            {"decay",".95"},
            {"dimensions","3"},
            {"surfaces_opacity","0"},
            {"points_opacity","0"},
            {"physics_multiplier","200"},
            {"mirror_force",".005"},
            {"flip_by_symmetry","0"},
        });

        if(false && !ValidateC4Graph(g)) {
            cout << "Graph validation failed!" << endl;
            return;
        }

        cs.add_scene(gs, "gs");

        cs.stage_macroblock(SilenceBlock(.1), 1);
        cs.render_microblock();
        cs.stage_macroblock(SilenceBlock(2), 2);
        gs->state_manager.set("flip_by_symmetry", "0");
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"decay", ".6"}});
        cs.render_microblock();

        g.render_json("c4_full.js");

        cout << g.size() << " nodes" << endl;

    } catch (exception& e){
        movecache.WriteCache();
        fhourstonesCache.WriteCache();
    }
}
