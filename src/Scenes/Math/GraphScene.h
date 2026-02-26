#pragma once

#include "../Scene.h"
#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Graph.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <vector>

class GraphScene : public ThreeDimensionScene {
public:
    double curr_hash;
    double next_hash;
    std::vector<unsigned int> color_scheme;
    GraphScene(std::shared_ptr<Graph> g, bool surfaces_on, const double width = 1, const double height = 1);

    void graph_to_3d();

    virtual int get_edge_color(const Node& node, const Node& neighbor);

    const StateQuery populate_state_query() const override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    void update_surfaces();

    virtual Surface make_surface(Node node) const;

    // Override the default surface render routine to make all graph surfaces point at the camera
    void render_surface(const Surface& surface) override;

    bool surfaces_override_unsafe; // For really big graphs, you can permanently turn off node stuff. This happens in the constructor, but careful when handling manually.
    std::shared_ptr<Graph> graph;

protected:
    std::unordered_map<std::string, std::pair<Surface, std::shared_ptr<Scene>>> graph_surface_map;

private:
    int last_node_count;
};
