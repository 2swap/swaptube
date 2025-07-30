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
        shared_ptr<LatexScene> ls_size = make_shared<LatexScene>(latex_text("Node count: "+to_string(g.size())), 1, .2, .1);

        gs->state_manager.set({
            {"q1", "1"},
            {"qi", "<t> .2 * cos"},
            {"qj", "<t> .314 * sin"},
            {"qk", "0"},
            {"decay",".98"},
            {"dimensions","3.98"},
            {"surfaces_opacity","0"},
            {"points_opacity","0"},
            {"physics_multiplier","100"},
        });

        cs.add_scene(gs, "gs");
        cs.add_scene(ls_size, "ls_size", .1, .12);
        //if(!ValidateC4Graph(g)) return;

        cs.stage_macroblock(SilenceBlock(5), 10);
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"dimensions", "3"}, {"mirror_force", ".5"}});
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"dimensions", "3.98"}, {"mirror_force", "0"}});
        cs.render_microblock();
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"dimensions", "3"}, {"mirror_force", ".5"}});
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"dimensions", "3.98"}, {"mirror_force", "0"}});
        cs.render_microblock();
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"dimensions", "3"}, {"mirror_force", ".5"}});
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"decay", ".5"}});
        cs.render_microblock();
        cs.render_microblock();
        g.render_json("c4_full.js");

    } catch (exception& e){
        movecache.WriteCache();
        fhourstonesCache.WriteCache();
    }
}
