#include <unordered_map>
#include <string>
#include <stdint.h>
#include "DataObject.h"

struct NodeConfig {
    uint32_t color;
    float radius;
    std::string label;
    NodeConfig() : color(0xffffffff), radius(1.0f), label("") {}
};
struct EdgeConfig {
    uint32_t color;
    float thickness;
    std::string label;
    EdgeConfig() : color(0xffffffff), thickness(1.0f), label("") {}
};

class GraphDrawingConfig : public DataObject {
private:
    std::unordered_map<double, NodeConfig> node_configs;
    std::unordered_map<double, EdgeConfig> edge_configs;
public:
    uint32_t get_node_color(double nodeId) const;
    uint32_t get_edge_color(double to, double from) const;
    float get_node_radius(double nodeId) const;
    void tick(const StateReturn& state) { }
    void transition_node_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void transition_node_label(const TransitionType tt, const double hash, const std::string& new_label);
    void transition_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color);
    void transition_all_node_colors(const TransitionType tt, const uint32_t new_color);
    void transition_all_edge_colors(const TransitionType tt, const uint32_t new_color);
    void step_transition(const TransitionType tt);
};
