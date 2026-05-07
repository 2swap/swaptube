#include <unordered_map>
#include <string>
#include <stdint.h>
#include "DataObject.h"
#include "../Core/Color.h"
#include "../IO/Writer.h"

struct NodeConfig {
    uint32_t color;
    float radius;
    std::string label;
    float splash_radius;
    float splash_opacity;
    uint32_t target_color;
    float target_radius;
    std::string target_label;
    TransitionType radius_transition_type;
    TransitionType label_transition_type;
    TransitionType color_transition_type;
    bool color_fade;
    NodeConfig() : color(0xffffffff), radius(1.0f), label(""),
                   splash_radius(0.0f), splash_opacity(1.0f),
                   target_color(0xffffffff), target_radius(1.0f), target_label(""),
                   radius_transition_type(MICRO), label_transition_type(MICRO), color_transition_type(MICRO),
                     color_fade(false) {}
};
struct EdgeConfig {
    uint32_t color;
    std::string label;
    uint32_t target_color;
    std::string target_label;
    TransitionType color_transition_type;
    TransitionType label_transition_type;
    bool color_transition_direction; // false: from lower hash to higher hash, true: from higher hash to lower hash
    bool color_fade;
    EdgeConfig() :        color(0xffffffff),        label(""),
                   target_color(0xffffffff), target_label(""),
                   color_transition_type(MICRO), label_transition_type(MICRO),
                   color_transition_direction(false), color_fade(false) {}
};


struct EdgeRenderData {
    uint32_t pre_color;
    uint32_t post_color;
    std::string label;
    float label_size;
    float midpoint_fraction;
    bool direction; // false: from lower hash to higher hash, true: from higher hash to lower hash
    bool is_fading;
    uint32_t fade_color;
};

struct NodeRenderData {
    uint32_t color;
    float radius;
    std::string label;
    float label_size;
    float splash_radius;
    float splash_opacity;
};

class GraphDrawingConfig : public DataObject {
private:
    std::unordered_map<double, NodeConfig> node_configs;
    std::unordered_map<double, EdgeConfig> edge_configs;
    void transition_edge_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void fade_edge_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void set_edge_color(const double hash, const uint32_t new_color);

public:
    NodeRenderData get_node_render_data(const double node_id, const float macroblock_fraction, const float microblock_fraction) const;
    EdgeRenderData get_edge_render_data(double to, double from, const float macroblock_fraction, const float microblock_fraction) const;

    void add_node_if_missing(double node_id);
    void add_edge_if_missing(double from, double to);

    void tick(const StateReturn& state);
    void transition_node_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void       fade_node_color(const TransitionType tt, const double hash, const uint32_t new_color);
    void        set_node_color(const double hash, const uint32_t new_color);
    void        set_all_node_colors(const uint32_t new_color);
    void        set_node_radius(const double hash, const float new_radius);
    void splash_node(const double hash);
    void transition_node_label(const TransitionType tt, const double hash, const std::string& new_label);
    void        set_node_label(const double hash, const string& new_label);
    void transition_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color);
    void        set_edge_color(const double hash1, const double hash2, const uint32_t new_color);
    void       fade_edge_color(const TransitionType tt, const double hash1, const double hash2, const uint32_t new_color);
    void transition_edge_label(const TransitionType tt, const double hash1, const double hash2, const std::string& new_label);
    void transition_all_node_colors(const TransitionType tt, const uint32_t new_color);
    void       fade_all_node_colors(const TransitionType tt, const uint32_t new_color);
    void       fade_all_edge_colors(const TransitionType tt, const uint32_t new_color);
    void        set_all_edge_colors(const uint32_t new_color);
    void step_transition(const TransitionType tt);
};
