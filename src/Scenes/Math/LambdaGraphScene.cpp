#pragma once

#include "GraphScene.cpp"
#include "LambdaScene.cpp"
#include "../../DataObjects/HashableString.cpp"

class LambdaGraphScene : public GraphScene<HashableString> {
public:
    LambdaGraphScene(Graph<HashableString>* g, string rep, const double width = 1, const double height = 1)
    : GraphScene(g, width, height), root_node_representation(rep) {
        HashableString* reppy = new HashableString(rep);
        graph->add_to_stack(reppy);
        state_manager.set(unordered_map<string, string>{
            {"physics_multiplier", "1"},
        });
    }

    Surface make_surface(Node<T> node) const override {
        shared_ptr<LambdaExpression> lambda = parse_lambda_from_string(node.data->representation);
        lambda->set_color_recursive(node.color);
        return Surface(glm::vec3(node.position),glm::vec3(1,0,0),glm::vec3(0,1,0), make_shared<LambdaScene>(lambda, .25, .25), node.opacity);
    }

private:
    string root_node_representation;
};
