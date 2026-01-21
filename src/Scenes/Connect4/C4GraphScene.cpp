#pragma once

#include "../Math/GraphScene.cpp"
#include "C4Scene.cpp"
#include "../../DataObjects/Connect4/C4Board.cpp"

class C4GraphScene : public GraphScene {
public:
    C4GraphScene(shared_ptr<Graph> g, bool surfaces_on, const string& rep, const C4BranchMode mode, const double width = 1, const double height = 1)
    : GraphScene(g, surfaces_on, width, height), root_node_representation(rep) {
        manager.set("physics_multiplier", "5");
        manager.set("decay", "0.5");

        if(mode == TRIM_STEADY_STATES){
            graph_to_check_if_points_are_in = graph;
        }

        C4Board* board;
        if(mode == SIMPLE_WEAK){
            shared_ptr<SteadyState> ss = find_steady_state(root_node_representation, nullptr, true);
            if(ss == NULL)
                throw runtime_error("No steady state found when making a SIMPLE_WEAK C4GraphScene.");
            board = new C4Board(mode, root_node_representation, ss);
        } else {
            board = new C4Board(mode, root_node_representation);
        }
        graph->add_to_stack(board);
    }

    int get_edge_color(const Node& node, const Node& neighbor){
        return min(node.data->representation.size(), neighbor.data->representation.size())%2==0 ? C4_RED : C4_YELLOW;
    }

private:
    string root_node_representation;
};
