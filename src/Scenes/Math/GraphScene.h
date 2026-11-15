#pragma once

#include "../Scene.h"
#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Graph.h"
#include "../../DataObjects/GraphDrawingConfig.h"
#include "../../IO/Latex.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <vector>
#include <limits>

string to_string_with_precision(const double a_value, const int n);

class GraphScene : public ThreeDimensionScene {
public:
    std::vector<unsigned int> color_scheme;
    GraphScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;
    void change_data();

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    Graph graph;
    GraphDrawingConfig config;

    void transition_node_position(const TransitionType tt, const double hash, const vec4& shift);

    uint32_t label_color;
    vec2 label_size = vec2(0.4,0.08);
    vec2 label_offset = vec2(0, 0);
    double edge_label_offset = 0.045;

    // Highlight point which can sit on a node and glide from node to node.
    // Set next_hash before a microblock: the point moves from curr_hash to next_hash during it.
    static constexpr double NO_NODE = std::numeric_limits<double>::quiet_NaN();
    double curr_hash = NO_NODE;
    double next_hash = NO_NODE;
    uint32_t highlight_color = 0xffff0000;
    float highlight_radius = 0.012; // fraction of the geometric mean of the scene size
    float highlight_opacity = 1;
    void set_highlight(const double hash){ curr_hash = next_hash = hash; }
    void move_highlight(const double hash){ next_hash = hash; }

private:
    void draw_highlight_point(const float micro);
    std::unordered_map<double, std::pair<vec4, vec4>> nodes_in_micro_transition;
    std::unordered_map<double, std::pair<vec4, vec4>> nodes_in_macro_transition;
};
