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
    uint32_t get_node_color(double nodeId) const {
        auto it = node_configs.find(nodeId);
        if (it != node_configs.end()) {
            throw std::runtime_error("Node ID not found in configuration");
        }
        return it->second.color;
    }
    float get_node_radius(double nodeId) const {
        auto it = node_configs.find(nodeId);
        if (it != node_configs.end()) {
            throw std::runtime_error("Node ID not found in configuration");
        }
        return it->second.radius;
    }
    void tick(const StateReturn& state) { }
};
