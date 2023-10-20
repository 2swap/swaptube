#pragma once

#include "scene.cpp"
#include "graph_scene.cpp"
#include "../../../Klotski/C4Board.cpp"

class C4GraphScene : public GraphScene<C4Board> {
public:
    C4GraphScene(const int width, const int height, Graph<C4Board>* g) : GraphScene(width, height, g) {init_surfaces();}
    C4GraphScene(Graph<C4Board>* g) : GraphScene(g) {init_surfaces();}

    void init_surfaces(){
        for(pair<double, Node<C4Board>> p : graph->nodes){
            Node<C4Board> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.x, node.y, node.z);
            surfaces.push_back(Surface(glm::vec3(0,0,0),glm::vec3(-3,0,0),glm::vec3(0,-3,0), new C4Scene(300, 300, node.data->representation)));
        }
    }

    void update_surfaces(){
        int i = 0;
        for(Surface& surface : surfaces){
            surface.center = points[i].position;
            i++;
        }
        rendered = false;
    }

    void query(bool& done_scene, Pixels*& p) override {
        if(do_physics){
            graph->iterate_physics(1, true);
            graph_to_3d();
            update_surfaces();
        }
        ThreeDimensionScene::query(done_scene, p);
    }

    ~C4GraphScene(){
        for (Surface surface : surfaces){
            delete surface.scenePointer;
        }
    }
};
