#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

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
        SAVE_FRAME_PNGS = false;

        Graph g;
        string variation = "";
        C4GraphScene gs(&g, false, variation, TRIM_STEADY_STATES);

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

        if(false && !ValidateC4Graph(g)) {
            cout << "Graph validation failed!" << endl;
            return;
        }

        gs.stage_macroblock(SilenceBlock(.3), 1);
        gs.render_microblock();
        gs.stage_macroblock(SilenceBlock(.5), 1);
        gs.manager.set("flip_by_symmetry", "0");
        gs.render_microblock();
        gs.stage_macroblock(SilenceBlock(.5), 1);
        gs.manager.transition(MICRO, "decay", ".6");
        gs.render_microblock();

        g.render_json("c4_full.js");

        cout << g.size() << " nodes" << endl;

    } catch (exception& e){
        cout << "Exception: " << e.what() << endl;
        movecache.WriteCache();
        fhourstonesCache.WriteCache();
    }
}
