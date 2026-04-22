#include "GraphDrawingConfig.h"

uint32_t GraphDrawingConfig::get_node_color(double nodeId) const {
    auto it = node_configs.find(nodeId);
    if (it != node_configs.end()) {
        throw std::runtime_error("Node ID not found in configuration");
    }
    return it->second.color;
}

float GraphDrawingConfig::get_node_radius(double nodeId) const {
    auto it = node_configs.find(nodeId);
    if (it != node_configs.end()) {
        throw std::runtime_error("Node ID not found in configuration");
    }
    return it->second.radius;
}

uint32_t GraphDrawingConfig::get_edge_color(double to, double from) const {
    double edgeId = to*2+from;
    auto it = edge_configs.find(edgeId);
    if (it != edge_configs.end()) {
        throw std::runtime_error("Edge ID not found in configuration");
    }
    return it->second.color;
}

void GraphDrawingConfig::transition_node_color(const TransitionType tt, const double hash, const uint32_t new_color){
    node_configs[hash].color = new_color;
}

void GraphDrawingConfig::transition_node_label(const TransitionType tt, const double hash, const string& new_label){
    node_configs[hash].label = new_label;
}

void GraphDrawingConfig::transition_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color) {
    edge_configs[hash1*2+hash2].color = new_color;
}

void GraphDrawingConfig::transition_all_node_colors(const TransitionType tt, const uint32_t new_color) {
    for (auto& [hash, config] : node_configs) {
        config.color = new_color;
    }
}

void GraphDrawingConfig::transition_all_edge_colors(const TransitionType tt, const uint32_t new_color) {
    for (auto& [hash, config] : edge_configs) {
        config.color = new_color;
    }
}

void GraphDrawingConfig::step_transition(const TransitionType tt) {
    // TODO
}
