#pragma once

#include "../Scene.h"
#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Graph.h"
#include "../../DataObjects/GraphDrawingConfig.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <vector>

string to_string_with_precision(const double a_value, const int n);

class GraphScene : public ThreeDimensionScene {
public:
    double curr_hash;
    double next_hash;
    std::vector<unsigned int> color_scheme;
    GraphScene(std::shared_ptr<Graph> g, const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    std::shared_ptr<Graph> graph;
    std::shared_ptr<GraphDrawingConfig> config;

    void transition_node_position(const TransitionType tt, const double hash, const vec4& shift);

private:
    int last_node_count;
    std::unordered_map<double, std::pair<vec4, vec4>> nodes_in_micro_transition;
    std::unordered_map<double, std::pair<vec4, vec4>> nodes_in_macro_transition;
};
