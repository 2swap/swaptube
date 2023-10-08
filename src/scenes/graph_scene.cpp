#pragma once

#include "scene.cpp"
#include "../../../Klotski/Graph.cpp" // I am too lazy to figure the right pattern out right now, so I'll assume you have my Klotski repo checked out in the same folder as this repo.

template <typename T>
class GraphScene : public ThreeDimensionScene {
public:
    GraphScene(const int width, const int height, Graph<T>* g) : ThreeDimensionScene(width, height), graph(g) {}
    GraphScene(Graph<T>* g) : ThreeDimensionScene(), graph(g) {}

    void graph_to_3d(){
        points.clear();
        lines.clear();
        for(pair<double, Node<T>> p : graph->nodes){
            double id = p.first;
            Node<T> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.x, node.y, node.z);
            points.push_back(node_pos);
            for(double neighbor_id : node.neighbors){
                Node<T> neighbor = graph->nodes.find(neighbor_id)->second;
                glm::vec3 neighbor_pos = glm::vec3(neighbor.x, neighbor.y, neighbor.z);
                lines.push_back(make_pair(node_pos, neighbor_pos));
            }
        }
        rendered = false;
    }

    Pixels* query(bool& done_scene) override {
        if(do_physics){
            graph->iterate_physics(8, false);
            graph_to_3d();
        }
        ThreeDimensionScene::query(done_scene);
    }

private:
    bool do_physics = true;
    Graph<T>* graph;
};
