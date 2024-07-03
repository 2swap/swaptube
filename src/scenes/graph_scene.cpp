#pragma once

#include "scene.cpp"
#include "Graph.cpp"

template <typename T>
class GraphScene : public ThreeDimensionScene {
public:
    GraphScene(const int width, const int height, Graph<T>* g) : ThreeDimensionScene(width, height), graph(g) {}
    GraphScene(Graph<T>* g) : ThreeDimensionScene(), graph(g) {}

    void graph_to_3d(){
        points.clear();
        lines.clear();

        for(pair<double, Node<T>> p : graph->nodes){
            Node<T> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.x, node.y, node.z);
            points.push_back(Point(node.data->representation, node_pos, WHITE));

            for(const Edge& neighbor_edge : node.neighbors){
                double neighbor_id = neighbor_edge.to;
                Node<T> neighbor = graph->nodes.find(neighbor_id)->second;
                glm::vec3 neighbor_pos = glm::vec3(neighbor.x, neighbor.y, neighbor.z);
                lines.push_back(Line(node_pos, neighbor_pos, get_edge_color(node, neighbor), neighbor_edge.opacity));
            }
        }
        rendered = false;
    }

    virtual int get_edge_color(const Node<T>& node, const Node<T>& neighbor){
        return WHITE;
    }

    virtual void inheritable_preprocessing(){}
    virtual void inheritable_postprocessing(){}

    void query(bool& done_scene, Pixels*& p) override {
        if(do_physics){
            graph->iterate_physics(physics_multiplier);
            graph_to_3d();
            inheritable_preprocessing();
        }
        ThreeDimensionScene::query(done_scene, p);
        inheritable_postprocessing();
    }
    
    int physics_multiplier = 1;

protected:
    bool do_physics = true;
    Graph<T>* graph;
};
