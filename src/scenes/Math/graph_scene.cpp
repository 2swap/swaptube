#pragma once

#include "../Scene.cpp"
#include "../Common/3DScene.cpp"
#include "Graph.cpp"

template <typename T>
class GraphScene : public ThreeDimensionScene {
public:
    GraphScene(Graph<T>* g, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : ThreeDimensionScene(width, height), graph(g) {}

    void graph_to_3d(){
        points.clear();
        lines.clear();

        for(pair<double, Node<T>> p : graph->nodes){
            Node<T> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.position);
            NodeHighlightType highlight = (node.data->get_hash() == graph->root_node_hash) ? BULLSEYE : (node.highlight? RING : NORMAL);
            points.push_back(Point(node_pos, node.color, highlight, node.opacity));

            for(const Edge& neighbor_edge : node.neighbors){
                double neighbor_id = neighbor_edge.to;
                Node<T> neighbor = graph->nodes.find(neighbor_id)->second;
                glm::vec3 neighbor_pos = glm::vec3(neighbor.position);
                lines.push_back(Line(node_pos, neighbor_pos, get_edge_color(node, neighbor), neighbor_edge.opacity));
            }
        }
        rendered = false;
    }

    virtual int get_edge_color(const Node<T>& node, const Node<T>& neighbor){
        return OPAQUE_WHITE;
    }

    virtual void inheritable_preprocessing(){}
    virtual void inheritable_postprocessing(){}

    void query(Pixels*& p) override {
        if(do_physics){
            graph->iterate_physics(physics_multiplier);
            graph_to_3d();
            inheritable_preprocessing();
        }
        ThreeDimensionScene::query(p);
        inheritable_postprocessing();
    }
    
    int physics_multiplier = 1;

protected:
    bool do_physics = true;
    Graph<T>* graph;
};
