#pragma once

#include "scene.cpp"
#include "graph_scene.cpp"
#include "c4_scene.cpp"
#include "../../../Klotski/C4Board.cpp"

class C4GraphScene : public GraphScene<C4Board> {
public:
    C4GraphScene(const int width, const int height, Graph<C4Board>* g, string rep, C4BranchMode mode) : GraphScene(width, height, g), root_node_representation(rep) {init_c4_graph(mode);}
    C4GraphScene(Graph<C4Board>* g, string rep, C4BranchMode mode) : GraphScene(g), root_node_representation(rep) {init_c4_graph(mode);}

    void init_c4_graph(C4BranchMode mode){
        c4_branch_mode = mode;

        if(mode != MANUAL){
            graph->add_to_stack(new C4Board(root_node_representation));
            graph->expand_graph_dfs();
            graph->make_edges_bidirectional();
        } else {
            graph->add_node(new C4Board(root_node_representation));
        }

        if(mode == SIMPLE_WEAK){
            find_steady_state(root_node_representation, 30000, ss_simple_weak, false);
        }
    }

    int get_edge_color(const Node<C4Board>& node, const Node<C4Board>& neighbor){
        return min(node.data->representation.size(), neighbor.data->representation.size())%2==0 ? C4_RED : C4_YELLOW;
    }

    void update_surfaces(){
        for(pair<double, Node<C4Board>> p : graph->nodes){
            Node<C4Board> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.x, node.y, node.z);
            C4Scene* sc = new C4Scene(600, 600, node.data->representation);
            surfaces.push_back(Surface("s" + node.data->representation, node_pos,glm::vec3(1,0,0),glm::vec3(0,1,0), sc));
        }
        rendered = false;
    }

    void clear_surfaces(){
        for(Surface s : surfaces){
            delete s.scenePointer;
        }
        surfaces.clear();
        rendered = false;
    }

    void inheritable_preprocessing() override{
        update_surfaces();
    }

    void inheritable_postprocessing() override{
        clear_surfaces();
    }

    void render_surface(const Surface& surface, int padcol) {
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
            surface.name,
            surface.center,
            glm::vec3(rotated_left_quat.x, rotated_left_quat.y, rotated_left_quat.z),
            glm::vec3(rotated_up_quat.x, rotated_up_quat.y, rotated_up_quat.z),
            surface.scenePointer
        );
        surface_rotated.opacity = surface.opacity;

        ThreeDimensionScene::render_surface(surface_rotated, padcol);
    }

    ~C4GraphScene(){
        for (Surface surface : surfaces){
            delete surface.scenePointer;
        }
    }

private:
    string root_node_representation;
};
