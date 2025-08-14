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
            //{"q1", "1"},
            //{"qi", "<t> .2 * cos"},
            //{"qj", "<t> .314 * sin"},
            //{"qk", "0"},
            {"decay",".99"},
            {"dimensions","3"},
            {"surfaces_opacity","0"},
            {"points_opacity","0"},
            {"physics_multiplier","200"},
            {"mirror_force",".005"},
            {"flip_by_symmetry","1"},
        });

        // Iterate over all graph nodes and search for steadystates
        int count = 0;
        if(false) for (auto& pair : g.nodes) {
            count++;
            cout << "Processed " << count << " nodes." << endl;
            auto& node = pair.second;
            string node_rep = node.data->representation;
            // Check if this node has no steady state, but a child's child does
            for (auto& edge1 : node.neighbors) {
                bool breakout = false;
                for(auto& edge : g.nodes.find(edge1.to)->second.neighbors) {
                    shared_ptr<SteadyState> ss = dynamic_cast<C4Board*>(g.nodes.find(edge.to)->second.data)->steadystate;
                    if (ss != nullptr) {
                        if (dynamic_cast<C4Board*>(node.data)->steadystate == nullptr) {
                            cout << "Node " << node_rep << " has no steady state, but child node does." << endl;
                            if(find_steady_state(node_rep, ss, false, true, 100, 100)) {
                                cout << "  Found steady state!" << endl;
                                breakout = true;
                                break;
                            }
                        }
                    }
                }
                if (breakout) {
                    break;
                }
            }
        }

        if(true && !ValidateC4Graph(g)) {
            cout << "Graph validation failed!" << endl;
            return;
        }

        cs.add_scene(gs, "gs");

        cs.stage_macroblock(SilenceBlock(.3), 1);
        cs.render_microblock();
        cs.stage_macroblock(SilenceBlock(2), 2);
        gs->state_manager.set("flip_by_symmetry", "0");
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"decay", ".6"}});
        cs.render_microblock();

        g.render_json("c4_full.js");

        cout << g.size() << " nodes" << endl;

    } catch (exception& e){
        cout << "Exception: " << e.what() << endl;
        movecache.WriteCache();
        fhourstonesCache.WriteCache();
    }
}
