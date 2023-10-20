#pragma once

#include "scene.cpp"
#include "../../../Klotski/Graph.cpp" // I am too lazy to figure the right pattern out right now, so I'll assume you have my Klotski repo checked out in the same folder as this repo.

template <typename T>
class C4GraphScene : public GraphScene {
public:
    C4GraphScene(const int width, const int height, Graph<T>* g) : ThreeDimensionScene(width, height), graph(g) {init();}
    C4GraphScene(Graph<T>* g) : ThreeDimensionScene(), graph(g) {init();}

    void init(){
        for(pair<double, Node<T>> p : graph->nodes){
            Node<T> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.x, node.y, node.z);
            surfaces.push_back(Surface(glm::vec3(0,0,0),glm::vec3(-3,0,0),glm::vec3(0,-3,0), new C4Scene(300, 300, node.data->representation)));
        }
    }

    void graph_to_3d(){
        points.clear();
        lines.clear();

        int i = 0;
        for(pair<double, Node<T>> p : graph->nodes){
            Node<T> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.x, node.y, node.z);
            surfaces[i].center = node_pos;
            i++;
            points.push_back(Point(node_pos, WHITE));
            
            for(double neighbor_id : node.neighbors){
                Node<T> neighbor = graph->nodes.find(neighbor_id)->second;
                glm::vec3 neighbor_pos = glm::vec3(neighbor.x, neighbor.y, neighbor.z);
                lines.push_back(Line(node_pos, neighbor_pos, WHITE));
            }
        }
        rendered = false;
    }

    void query(bool& done_scene, Pixels*& p) override {
        if(do_physics){
            graph->iterate_physics(1, true);
            graph_to_3d();
        }
        ThreeDimensionScene::query(done_scene, p);
    }

    ~GraphScene(){
        for (Surface surface : surfaces){
            delete surface.scenePointer;
        }
    }

private:
    bool do_physics = true;
    Graph<T>* graph;
};
