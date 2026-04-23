#include <unordered_map>
#include <string>
#include <stdint.h>
#include "DataObject.h"

struct NodeConfig {
    uint32_t color;
    float radius;
    std::string label;
    uint32_t target_color;
    float target_radius;
    std::string target_label;
    TransitionType color_transition_type;
    TransitionType radius_transition_type;
    TransitionType label_transition_type;
    NodeConfig() :        color(0xffffffff),        radius(1.0f),        label(""),
                   target_color(0xffffffff), target_radius(1.0f), target_label(""),
                   color_transition_type(MICRO), radius_transition_type(MICRO), label_transition_type(MICRO) {}
};
struct EdgeConfig {
    uint32_t color;
    std::string label;
    uint32_t target_color;
    std::string target_label;
    TransitionType color_transition_type;
    TransitionType label_transition_type;
    EdgeConfig() :        color(0xffffffff),        label(""),
                   target_color(0xffffffff), target_label(""),
                   color_transition_type(MICRO), label_transition_type(MICRO) {}
};

class GraphDrawingConfig : public DataObject {
private:
    std::unordered_map<double, NodeConfig> node_configs;
    std::unordered_map<double, EdgeConfig> edge_configs;
public:
    void add_node_if_missing(double node_id);
    void add_edge_if_missing(double from, double to);
    uint32_t get_node_color(double node_id, const float macroblock_fraction, const float microblock_fraction) const;
    float get_node_radius(double node_id, const float macroblock_fraction, const float microblock_fraction) const;
    string get_node_label(double node_id, const float macroblock_fraction, const float microblock_fraction) const;
    float get_node_label_size(double node_id, const float macroblock_fraction, const float microblock_fraction) const;
    uint32_t get_edge_color(double to, double from, const float macroblock_fraction, const float microblock_fraction) const;
    void tick(const StateReturn& state) { }
    void transition_node_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void transition_node_label(const TransitionType tt, const double hash, const std::string& new_label);
    void transition_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color);
    void transition_edge_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void transition_edge_label(const TransitionType tt, const double hash1, const double hash2, const std::string& new_label);
    void transition_all_node_colors(const TransitionType tt, const uint32_t new_color);
    void transition_all_edge_colors(const TransitionType tt, const uint32_t new_color);
    void step_transition(const TransitionType tt);
};
