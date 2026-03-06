#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Connect4/C4Scene.h"
#include "../Scenes/Connect4/C4GraphScene.h"
#include "../DataObjects/Connect4/TreeValidator.h"
#include "../DataObjects/Connect4/SteadyState.h"

void find_steadystates_by_children(const Graph& g, Node& node) {
    /*
    for (auto& edge1 : node.neighbors) {
        for(auto& edge : g.nodes.find(edge1.to)->second.neighbors) {
            Node child_node = g.nodes.find(edge.to)->second;
            C4Board* child = dynamic_cast<C4Board*>(child_node.data);
            if (child->representation.size() <= node.data->representation.size()) continue;
            // If the child has a neighbor whose representation is not longer than its own, return
            for (auto& edge2 : child_node.neighbors) {
                Node grandchild_node = g.nodes.find(edge2.to)->second;
                C4Board* grandchild = dynamic_cast<C4Board*>(grandchild_node.data);
                if (grandchild->representation.size() > child->representation.size()) {
                    //return;
                }
            }
        }
    }
    */
    for (auto& edge1 : node.neighbors) {
        for(auto& edge2 : g.nodes.find(edge1.to)->second.neighbors) {
            for (auto& edge3 : g.nodes.find(edge2.to)->second.neighbors) {
                for(auto& edge4 : g.nodes.find(edge3.to)->second.neighbors) {
                    Node child_node = g.nodes.find(edge4.to)->second;
                    C4Board* child = dynamic_cast<C4Board*>(child_node.data);
                    //if (child->representation.size() <= node.data->representation.size()) continue;
                    cout << "Attempting backpropagation of: " << child->representation << endl;
                    shared_ptr<SteadyState> ss = child->steadystate;
                    if (ss != nullptr) {
                        if(find_steady_state(node.data->representation, nullptr, false, true, 50, 50)) {
                            cout << "  Found steady state!" << endl;
                            return;
                        }
                        if(find_steady_state(node.data->representation, ss, false, true, 50, 50)) {
                            cout << "  Found steady state!" << endl;
                            return;
                        }
                    }
                }
            }
        }
    }
}
void render_video() {
    try{
        cout << "Building graph..." << endl;
        shared_ptr<Graph> g = make_shared<Graph>();
        string variation = "";
        C4GraphScene gs(g, false, variation, TRIM_STEADY_STATES);
        cout << "Expanding graph..." << endl;
        g->expand(-1);
        g->make_bidirectional();
        cout << "Graph built with " << g->size() << " nodes." << endl;

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
            {"flip_by_symmetry","1"},
        });

        /*
        // Iterate over all graph nodes and search for steadystates
        int count = 0;

        movecache.WriteCache();
        if(false) for (auto& pair : g.nodes) {
            count++;
            auto& node = pair.second;
            string node_rep = node.data->representation;
            shared_ptr<SteadyState> ss = dynamic_cast<C4Board*>(node.data)->steadystate;
            if(ss != nullptr) continue;
            cout << "Processed " << count << " nodes." << endl;
            find_steadystates_by_children(g, node);
        }
        movecache.WriteCache();
        */

        cout << "Validating graph..." << endl;
        if(!ValidateC4Graph(*g)) {
            cout << "Graph validation failed!" << endl;
            return;
        }

        stage_macroblock(SilenceBlock(.3), 1);
        gs.render_microblock();
        stage_macroblock(SilenceBlock(.5), 1);
        gs.manager.set("flip_by_symmetry", "0");
        gs.render_microblock();
        stage_macroblock(SilenceBlock(.5), 1);
        gs.manager.transition(MICRO, "decay", ".6");
        gs.render_microblock();

        g->render_json("c4_full.js");

        cout << g->size() << " nodes" << endl;

    } catch (exception& e){
        cout << "Exception: " << e.what() << endl;
        //movecache.WriteCache();
        //fhourstonesCache.WriteCache();
    }
}
