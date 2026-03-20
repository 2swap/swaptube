#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Connect4/C4Scene.h"
#include "../Scenes/Connect4/C4GraphScene.h"
#include "../DataObjects/Connect4/TreeValidator.h"
#include "../DataObjects/Connect4/SteadyState.h"

void render_video() {
    /*
    string variation = "44444456233656626";
    C4Board b(TRIM_STEADY_STATES, variation);
    for(int i = 1; i <= 7; i++) {
        if(!b.is_legal(i)) continue;
        shared_ptr<SteadyState> ss = find_steady_state(variation + to_string(i), nullptr, true, false, 100, 100);
        if(ss == nullptr) {
            cout << "Steady state not found for variation " << variation + to_string(i) << "!" << endl;
            return;
        }
    }
    */
    try{
        cout << "Building graph..." << endl;
        shared_ptr<Graph> g = make_shared<Graph>();
        string variation = "";
        C4GraphScene gs(g, false, variation, TRIM_STEADY_STATES);

        gs.manager.set({
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "0"},
            {"decay",".98"},
            {"dimensions","3"},
            {"surfaces_opacity","0"},
            {"points_opacity","0"},
            {"physics_multiplier","200"},
            {"mirror_force",".005"},
            {"desired_nodes","0"},
        });

        stage_macroblock(SilenceBlock(4), 4);
        gs.manager.transition(MICRO, "desired_nodes", "9000");
        gs.render_microblock();
        cout << endl << endl << g->size() << " nodes" << endl << endl;
        gs.manager.transition(MICRO, "decay", ".95");
        gs.manager.transition(MICRO, "physics_multiplier", "100");
        gs.render_microblock();
        gs.manager.transition(MICRO, "decay", ".7");
        gs.render_microblock();
        gs.manager.transition(MICRO, "decay", ".3");
        gs.render_microblock();

        cout << endl << endl << g->size() << " nodes" << endl << endl;
        cout << "Validating graph..." << endl;
        if(!ValidateC4Graph(*g)) {
            cout << "Graph validation failed!" << endl;
            return;
        }

        g->render_json("../../WeakC4/c4_full.js");

        cout << endl << endl << g->size() << " nodes" << endl << endl;

    } catch (exception& e){
        cout << "Exception: " << e.what() << endl;
        get_movecache().WriteCache();
        //get_fhourstonescache().WriteCache();
    }
}
