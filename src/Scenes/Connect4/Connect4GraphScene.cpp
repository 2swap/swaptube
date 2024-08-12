#pragma once

#include "../Math/GraphScene.cpp"
#include "Connect4Scene.cpp"
#include "../../DataObjects/Connect4/C4Board.cpp"

class C4GraphScene : public GraphScene<C4Board> {
public:
    C4GraphScene(Graph<C4Board>* g, string rep, C4BranchMode mode, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : GraphScene(g, width, height), root_node_representation(rep) {
        c4_branch_mode = mode;

        if(mode == TRIM_STEADY_STATES){
            graph_to_check_if_points_are_in = graph;
        }

        C4Board* board;
        if(mode == SIMPLE_WEAK){
            shared_ptr<SteadyState> ss = find_steady_state(root_node_representation, 30000);
            if(ss == NULL)
                failout("No steady state found when making a SIMPLE_WEAK C4GraphScene.");
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

    void update_surfaces(){
        for(pair<double, Node<C4Board>> p : graph->nodes){
            Node<C4Board> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.position);
            add_surface(Surface(node_pos,glm::vec3(1,0,0),glm::vec3(0,1,0), make_shared<C4Scene>(node.data->representation, 600, 600)));
        }
    }

    void inheritable_preprocessing() override{
        update_surfaces();
    }

    void inheritable_postprocessing() override{
        clear_surfaces();
    }

    void render_surface(const Surface& surface) {
        //make all the boards face the camera
        glm::quat conj2 = glm::conjugate(camera_direction) * glm::conjugate(camera_direction);
        glm::quat cam2 = camera_direction * camera_direction;

        // Rotate pos_x_dir vector
        glm::quat left_as_quat(0.0f, surface.pos_x_dir.x, surface.pos_x_dir.y, surface.pos_x_dir.z);
        glm::quat rotated_left_quat = conj2 * left_as_quat * cam2;

        // Rotate pos_y_dir vector
        glm::quat up_as_quat(0.0f, surface.pos_y_dir.x, surface.pos_y_dir.y, surface.pos_y_dir.z);
        glm::quat rotated_up_quat = conj2 * up_as_quat * cam2;

        Surface surface_rotated(
            surface.center,
            glm::vec3(rotated_left_quat.x, rotated_left_quat.y, rotated_left_quat.z),
            glm::vec3(rotated_up_quat.x, rotated_up_quat.y, rotated_up_quat.z),
            surface.scenePointer
        );

        ThreeDimensionScene::render_surface(surface_rotated);
    }

    bool color_edges = true;

private:
    string root_node_representation;
};
