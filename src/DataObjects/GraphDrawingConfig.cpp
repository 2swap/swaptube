#include "GraphDrawingConfig.h"

NodeRenderData GraphDrawingConfig::get_node_render_data(const double node_id, const float macroblock_fraction, const float microblock_fraction) const {
    auto it = node_configs.find(node_id);
    NodeRenderData data;

    // Color
    if(it->second.color_fade) {
        TransitionType ctt = it->second.color_transition_type;
        float color_fraction = ctt == MICRO ? microblock_fraction : macroblock_fraction;
        data.color = colorlerp(it->second.color, it->second.target_color, color_fraction);
    }
    else data.color = it->second.color;

    // Radius
    TransitionType rtt = it->second.radius_transition_type;
    float radius_fraction = rtt == MICRO ? microblock_fraction : macroblock_fraction;
    data.radius = smoothlerp(it->second.radius, it->second.target_radius, radius_fraction);

    // Label
    TransitionType ltt = it->second.label_transition_type;
    float label_fraction = ltt == MICRO ? microblock_fraction : macroblock_fraction;
    data.label = label_fraction < 0.5f ? it->second.label : it->second.target_label;
    if (it->second.label == it->second.target_label) data.label_size = 1.0f;
    else {
        float magic_parabola = -8*label_fraction*label_fraction + 14*label_fraction - 5;
        data.label_size = label_fraction < 0.5f ? (1-label_fraction*2) : magic_parabola;
    }

    // Splash
    data.splash_radius = it->second.splash_radius;
    data.splash_opacity = it->second.splash_opacity;
    return data;
}

EdgeRenderData GraphDrawingConfig::get_edge_render_data(double to, double from, const float macroblock_fraction, const float microblock_fraction) const {
    double edge_id = to*2+from;
    auto it = edge_configs.find(edge_id);
    EdgeRenderData data;

    // Color
    TransitionType ctt = it->second.color_transition_type;
    float color_fraction = ctt == MICRO ? microblock_fraction : macroblock_fraction;
    if(it->second.color_fade) {
        data.pre_color = data.post_color = colorlerp(it->second.color, it->second.target_color, color_fraction);
        data.is_fading = true;
    }
    else {
        data.pre_color = it->second.color;
        data.post_color = it->second.target_color;
        data.is_fading = false;
        data.midpoint_fraction = color_fraction;
    }

    // Label
    TransitionType ltt = it->second.label_transition_type;
    float label_fraction = ltt == MICRO ? microblock_fraction : macroblock_fraction;
    data.label = label_fraction < 0.5f ? it->second.label : it->second.target_label;

    if (it->second.label == it->second.target_label) data.label_size = 1.0f;
    else {
        float relevant_fraction = ltt == MICRO ? microblock_fraction : macroblock_fraction;
        float magic_parabola = -8*label_fraction*label_fraction + 14*label_fraction - 5;
        data.label_size = label_fraction < 0.5f ? (1-label_fraction*2) : magic_parabola;
    }
    if(it->second.label_splashing) {
        TransitionType lstt = it->second.label_splashing_type;
        float label_splashing_fraction = lstt == MICRO ? microblock_fraction : macroblock_fraction;
        float magic_quartic = 1.75-square(square(label_splashing_fraction*2-1))*.75;
        data.label_size *= magic_quartic;
    }

    // Direction
    data.direction = it->second.color_transition_direction;

    data.is_dashed = it->second.is_dashed;
    if(chill){
        // This is designed for 30fps. If framerate is different, adjust.
        int age_to_use = it->second.age * 30 / get_video_framerate_fps();
        data.post_color = data.pre_color = colorlerp(data.post_color, 0xffffffff, 1 / (1 + square(age_to_use) * .1));
    }

    return data;
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

void GraphDrawingConfig::set_node_color(const double hash, const uint32_t new_color) {
    node_configs[hash].target_color = new_color;
    node_configs[hash].color = new_color;
}

void GraphDrawingConfig::set_all_node_colors(const uint32_t new_color) {
    for (auto& [hash, config] : node_configs) {
        set_node_color(hash, new_color);
    }
}

void GraphDrawingConfig::set_node_radius(const double hash, const float new_radius) {
    node_configs[hash].target_radius = new_radius;
    node_configs[hash].radius = new_radius;
}

void GraphDrawingConfig::transition_node_radius(const TransitionType tt, const double hash, const float new_radius) {
    node_configs[hash].target_radius = new_radius;
    node_configs[hash].radius_transition_type = tt;
}

void GraphDrawingConfig::set_all_edge_colors(const uint32_t new_color) {
    for (auto& [hash, config] : edge_configs) {
        set_edge_color(hash, new_color);
    }
}

void GraphDrawingConfig::set_node_label(const double hash, const string& new_label){
    node_configs[hash].target_label = new_label;
    node_configs[hash].label = new_label;
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

void GraphDrawingConfig::set_edge_dashed(const double hash1, const double hash2, const bool is_dashed) {
    edge_configs[hash1*2+hash2].is_dashed = is_dashed;
    edge_configs[hash2*2+hash1].is_dashed = is_dashed;
}

void GraphDrawingConfig::set_edge_color(const double hash, const uint32_t new_color) {
    if (edge_configs[hash].color != new_color) {
        edge_configs[hash].age = 0;
    }
    if(new_color == 0xff00ffff) {
        edge_configs[hash].age = 1000;
    }
    edge_configs[hash].target_color = new_color;
    edge_configs[hash].color = new_color;
}

void GraphDrawingConfig::splash_node(const double hash) {
    auto& config = node_configs[hash];
    config.splash_radius = 0.0f;
    config.splash_opacity = 1.0f;
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

void GraphDrawingConfig::set_edge_color(const double hash1, const double hash2, const uint32_t new_color) {
    set_edge_color(hash1*2+hash2, new_color);
    set_edge_color(hash2*2+hash1, new_color);
}

void GraphDrawingConfig::set_edge_label(const double hash1, const double hash2, const string& new_label) {
    edge_configs[hash1*2+hash2].target_label = new_label;
    edge_configs[hash2*2+hash1].target_label = new_label;
    edge_configs[hash1*2+hash2].label = new_label;
    edge_configs[hash2*2+hash1].label = new_label;
}

void GraphDrawingConfig::transition_edge_label(const TransitionType tt, const double hash1, const double hash2, const string& new_label) {
    transition_edge_label(tt, hash1*2+hash2, new_label);
    transition_edge_label(tt, hash2*2+hash1, new_label);
}

void GraphDrawingConfig::transition_edge_label(const TransitionType tt, const double hash, const string& new_label) {
    edge_configs[hash].target_label = new_label;
    edge_configs[hash].label_transition_type = tt;
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
        edge_configs[edge_id].color_transition_direction = hash1 > hash2;
        edge_configs[edge_id].color_fade = false;
        edge_configs[edge_id].color = 0x00ffffff;
        edge_configs[edge_id].target_color = 0xffffffff;
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

void GraphDrawingConfig::transition_all_edge_labels(const TransitionType tt, const string& new_label) {
    for (auto& [hash, config] : edge_configs) {
        transition_edge_label(tt, hash, new_label);
    }
}

void GraphDrawingConfig::transition_all_node_radii(const TransitionType tt, const float new_radius) {
    for (auto& [hash, config] : node_configs) {
        transition_node_radius(tt, hash, new_radius);
    }
}

void GraphDrawingConfig::splash_edge_label(const TransitionType tt, const double hash) {
    edge_configs[hash].label_splashing = true;
    edge_configs[hash].label_splashing_type = tt;
}

void GraphDrawingConfig::splash_edge_label(const TransitionType tt, const double hash1, const double hash2) {
    splash_edge_label(tt, hash1*2+hash2);
    splash_edge_label(tt, hash2*2+hash1);
}

void GraphDrawingConfig::step_transition(const TransitionType tt) {
    for (auto& [hash, config] : node_configs) {
        if (config.radius_transition_type == tt) config.radius = config.target_radius;
        if (config. label_transition_type == tt) config.label = config.target_label;
        if (config. color_transition_type == tt) config.color = config.target_color;
    }
    for (auto& [hash, config] : edge_configs) {
        if (config.color_transition_type == tt) config.color = config.target_color;
        if (config.label_transition_type == tt) config.label = config.target_label;
        if (config.label_splashing_type == tt) config.label_splashing = false;
    }
}

void GraphDrawingConfig::tick(const StateReturn& state) {
    for (auto& [hash, config] : node_configs) {
        if(config.target_color != config.color && !config.color_fade) {
            config.splash_radius = 0.0f;
            config.splash_opacity = 1.0f;
            config.color = config.target_color;
        }
        int framerate = get_video_framerate_fps();
        for(int i = 0; i < 120; i += framerate) { // Designed for 60fps
            config.splash_radius = sqrt(config.splash_radius*config.splash_radius + .07);
            config.splash_opacity -= 0.012f;
        }
    }
    for (auto& [hash, config] : edge_configs) {
        config.age++;
    }
}
