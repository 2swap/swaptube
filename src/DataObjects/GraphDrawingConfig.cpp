#include "GraphDrawingConfig.h"
#include "../Core/Color.h"

float GraphDrawingConfig::get_node_splash_radius(double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    return it->second.splash_radius;
}

float GraphDrawingConfig::get_node_splash_opacity(double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    return it->second.splash_opacity;
}

uint32_t GraphDrawingConfig::get_node_color(double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    if(it->second.color_fade) {
        TransitionType tt = it->second.color_transition_type;
        float relevant_fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
        return colorlerp(it->second.color, it->second.target_color, relevant_fraction);
    }
    return it->second.color;
}

float GraphDrawingConfig::get_node_radius(double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    TransitionType tt = it->second.radius_transition_type;
    float relevant_fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
    return smoothlerp(it->second.radius, it->second.target_radius, relevant_fraction);
}

string GraphDrawingConfig::get_node_label(double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    TransitionType tt = it->second.label_transition_type;
    float relevant_fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
    return relevant_fraction < 0.5f ? it->second.label : it->second.target_label;
}

float GraphDrawingConfig::get_node_label_size(double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    if (it->second.label == it->second.target_label) return 1.0f;
    TransitionType tt = it->second.label_transition_type;
    float relevant_fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
    float magic_parabola = -12*relevant_fraction*relevant_fraction + 20*relevant_fraction - 7;
    return relevant_fraction < 0.5f ? (1-relevant_fraction*2) : magic_parabola;
}

uint32_t GraphDrawingConfig::get_edge_color(double to, double from, const float macroblock_fraction, const float microblock_fraction) const {
    double edge_id = to*2+from;
    auto it = edge_configs.find(edge_id);
    //TransitionType tt = it->second.color_transition_type;
    //float relevant_fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
    //return colorlerp(it->second.color, it->second.target_color, relevant_fraction);
    return it->second.color;
}

uint32_t GraphDrawingConfig::get_edge_fade_color(double to, double from, const float macroblock_fraction, const float microblock_fraction) const {
    double edge_id = to*2+from;
    auto it = edge_configs.find(edge_id);
    if(it->second.color_fade) {
        TransitionType tt = it->second.color_transition_type;
        float relevant_fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
        return colorlerp(it->second.color, it->second.target_color, relevant_fraction);
    } else {
        return it->second.color;
    }
}

uint32_t GraphDrawingConfig::get_edge_target_color(double to, double from, const float macroblock_fraction, const float microblock_fraction) const {
    double edge_id = to*2+from;
    auto it = edge_configs.find(edge_id);
    return it->second.target_color;
}

float GraphDrawingConfig::get_edge_midpoint_fraction(double to, double from, const float macroblock_fraction, const float microblock_fraction) const {
    double edge_id = to*2+from;
    auto it = edge_configs.find(edge_id);
    TransitionType tt = it->second.color_transition_type;
    float fraction = tt == MICRO ? microblock_fraction : macroblock_fraction;
    return fraction;
}

bool GraphDrawingConfig::get_edge_direction(double to, double from) const {
    double edge_id = to*2+from;
    auto it = edge_configs.find(edge_id);
    return it->second.color_transition_direction;
}

void GraphDrawingConfig::transition_node_color(const TransitionType tt, const double hash, const uint32_t new_color){
    node_configs[hash].target_color = new_color;
    node_configs[hash].color_transition_type = tt;
    node_configs[hash].color_fade = false;
}

void GraphDrawingConfig::fade_node_color(const TransitionType tt, const double hash, const uint32_t new_color){
    node_configs[hash].target_color = new_color;
    node_configs[hash].color_transition_type = tt;
    node_configs[hash].color_fade = true;
}

void GraphDrawingConfig::transition_node_label(const TransitionType tt, const double hash, const string& new_label){
    node_configs[hash].target_label = new_label;
    node_configs[hash].label_transition_type = tt;
}

void GraphDrawingConfig::transition_edge_color(const TransitionType tt, const double hash, const uint32_t new_color) {
    edge_configs[hash].target_color = new_color;
    edge_configs[hash].color_transition_type = tt;
    edge_configs[hash].color_fade = false;
}

void GraphDrawingConfig::fade_edge_color(const TransitionType tt, const double hash, const uint32_t new_color) {
    edge_configs[hash].target_color = new_color;
    edge_configs[hash].color_transition_type = tt;
    edge_configs[hash].color_fade = true;
}

void GraphDrawingConfig::transition_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color) {
    transition_edge_color(tt, hash1*2+hash2, new_color);
    transition_edge_color(tt, hash2*2+hash1, new_color);
    bool ctd = hash1 > hash2;
    edge_configs[hash1*2+hash2].color_transition_direction = ctd;
    edge_configs[hash2*2+hash1].color_transition_direction = ctd;
}

void GraphDrawingConfig::fade_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color) {
    fade_edge_color(tt, hash1*2+hash2, new_color);
    fade_edge_color(tt, hash2*2+hash1, new_color);
}

void GraphDrawingConfig::transition_edge_label(const TransitionType tt, const double hash1, const double hash2, const string& new_label) {
    edge_configs[hash1*2+hash2].target_label = new_label;
    edge_configs[hash2*2+hash1].target_label = new_label;
    edge_configs[hash1*2+hash2].label_transition_type = tt;
    edge_configs[hash2*2+hash1].label_transition_type = tt;
}

void GraphDrawingConfig::add_node_if_missing(const double hash) {
    if (node_configs.find(hash) == node_configs.end()) {
        node_configs[hash] = NodeConfig();
    }
}

void GraphDrawingConfig::add_edge_if_missing(const double hash1, const double hash2) {
    double edge_id = hash1*2+hash2;
    if (edge_configs.find(edge_id) == edge_configs.end()) {
        edge_configs[edge_id] = EdgeConfig();
    }
}

void GraphDrawingConfig::transition_all_node_colors(const TransitionType tt, const uint32_t new_color) {
    for (auto& [hash, config] : node_configs) {
        transition_node_color(tt, hash, new_color);
    }
}

void GraphDrawingConfig::fade_all_node_colors(const TransitionType tt, const uint32_t new_color) {
    for (auto& [hash, config] : node_configs) {
        fade_node_color(tt, hash, new_color);
    }
}

void GraphDrawingConfig::fade_all_edge_colors(const TransitionType tt, const uint32_t new_color) {
    for (auto& [hash, config] : edge_configs) {
        fade_edge_color(tt, hash, new_color);
    }
}

void GraphDrawingConfig::step_transition(const TransitionType tt) {
    for (auto& [hash, config] : node_configs) {
        if (config.radius_transition_type == tt) config.radius = config.target_radius;
        if (config. label_transition_type == tt) config.label = config.target_label;
    }
    for (auto& [hash, config] : edge_configs) {
        if (config.color_transition_type == tt) config.color = config.target_color;
        if (config.label_transition_type == tt) config.label = config.target_label;
    }
}

void GraphDrawingConfig::tick(const StateReturn& state) {
    for (auto& [hash, config] : node_configs) {
        if(config.target_color != config.color && !config.color_fade) {
            config.splash_radius = 0.0f;
            config.splash_opacity = 1.0f;
        }
        config.color = config.target_color;
        config.splash_radius = sqrt(config.splash_radius*config.splash_radius + .5);
        config.splash_opacity -= 0.025f;
    }
}
