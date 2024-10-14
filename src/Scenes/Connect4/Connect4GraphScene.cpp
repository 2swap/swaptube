#pragma once

#include "../Math/GraphScene.cpp"
#include "Connect4Scene.cpp"
#include "../../DataObjects/Connect4/C4Board.cpp"

class C4GraphScene : public GraphScene<C4Board> {
public:
    C4GraphScene(Graph<C4Board>* g, string rep, C4BranchMode mode, const double width = 1, const double height = 1)
    : GraphScene(g, width, height), root_node_representation(rep) {
        c4_branch_mode = mode;

        if(mode == TRIM_STEADY_STATES){
            graph_to_check_if_points_are_in = graph;
        }

        C4Board* board;
        if(mode == SIMPLE_WEAK){
            shared_ptr<SteadyState> ss = find_steady_state(root_node_representation, 30000);
            if(ss == NULL)
                throw runtime_exception("No steady state found when making a SIMPLE_WEAK C4GraphScene.");
            board = new C4Board(root_node_representation, ss);
        } else {
            board = new C4Board(root_node_representation);
        }
        graph->add_to_stack(board);

        if(mode != MANUAL){
            graph->expand_graph(false);
        }
        cout << "GRAPH SIZE: " << graph->size() << endl;
    }

    int get_edge_color(const Node<C4Board>& node, const Node<C4Board>& neighbor){
        if(!color_edges) return OPAQUE_WHITE;
        return min(node.data->representation.size(), neighbor.data->representation.size())%2==0 ? C4_RED : C4_YELLOW;
    }

    Surface make_surface(Node<T> node) const override {
        return Surface(glm::vec3(node.position),glm::vec3(1,0,0),glm::vec3(0,1,0), make_shared<C4Scene>(node.data->representation, 600, 600), node.opacity);
    }

    bool color_edges = true;

private:
    string root_node_representation;
};
