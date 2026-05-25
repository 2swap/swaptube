#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

// lat long map
unordered_map<string, vec2> netherlands_cities = {
    {"Amsterdam", vec2(52.3676, 4.9041)},
    {"The Hague", vec2(52.0705, 4.3007)},
    {"Rotterdam", vec2(51.9244, 4.4777)},
    {"Utrecht", vec2(52.0907, 5.1214)},
    {"Breda", vec2(51.5719, 4.7683)},
    {"Eindhoven", vec2(51.4416, 5.4697)},
    {"Arnhem", vec2(51.9851, 5.8987)},
    {"Zwolle", vec2(52.5168, 6.0830)},
    {"Emmen", vec2(52.7795, 6.9061)},
    {"Groningen", vec2(53.2194, 6.5665)},
    {"Leeuwarden", vec2(53.2012, 5.7999)},
    {"'s-Hertogenbosch", vec2(51.6978, 5.3037)},
    {"Tillburg", vec2(51.5555, 5.0913)},
    {"Meppel", vec2(52.7917, 6.1789)},
    {"Veenendaal", vec2(52.0231, 5.3889)},
    {"Almere", vec2(52.3508, 5.2647)},
};

vector<pair<string, string>> netherlands_edges_1 = {
    {"Rotterdam", "Utrecht"},
    {"Rotterdam", "Breda"},
    {"Breda", "Tillburg"},
    {"Tillburg", "Eindhoven"},
    {"Eindhoven", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Utrecht"},
    {"Veenendaal", "Arnhem"},
    {"Arnhem", "Zwolle"},
    {"Zwolle", "Emmen"},
    {"Zwolle", "Meppel"},
    {"Meppel", "Groningen"},
    {"Meppel", "Leeuwarden"},
    {"The Hague", "Rotterdam"},
    {"Amsterdam", "Utrecht"},
    {"Amsterdam", "Almere"},
    {"Utrecht", "Veenendaal"},
};

vector<pair<string, string>> netherlands_edges_2 = {
    {"Amsterdam", "The Hague"},
    {"Tillburg", "'s-Hertogenbosch"},
    {"Groningen", "Leeuwarden"},
    {"Almere", "Zwolle"},
};

vector<pair<string, string>> netherlands_edges_3 = {
    {"The Hague", "Utrecht"},
    {"Rotterdam", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Arnhem"},
};

void reset_graph(shared_ptr<GraphScene> gs, uint32_t col = 0xffffffff) {
    // Cleanup: transition all nodes and edges back to white and radius 1
    gs->config->fade_all_node_colors(MICRO, col);
    gs->config->fade_all_edge_colors(MICRO, col);
    gs->config->transition_all_node_radii(MICRO, 1);
}

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->label_color = 0xffffffff;
    gs->label_offset = vec2(0, 0.02);
    gs->edge_label_offset = 0.015;
    gs->label_size = vec2(0.4, 0.04);
    gs->manager.transition(MACRO, "globe_opacity", "1");
    vec2 center = (netherlands_cities["Utrecht"] + netherlands_cities["Zwolle"]) / 2.0;
    set_camera_to_lat_long(gs, center, true, MACRO);
    gs->manager.set("timey", "3"); // Hack
    gs->manager.set({
        {"physics_multiplier","0"},
        {"d", ".05"},
    });
    // Plot cities as nodes and roads as edges, expanding east->west
    stage_macroblock(SilenceBlock(1), 1);
    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        double hash = HashableString(city).get_hash();
        g->add_node(new HashableString(city));
        g->move_node(hash, position);
    }
    gs->render_microblock();

    // Load Netherlands map
    stage_macroblock(FileBlock("Here's a simplified graph of the Netherlands."), 1);
    gs->manager.transition(MICRO, "timey", "{t} 3 -");
    gs->render_microblock();

    // Plot cities as nodes and roads as edges, expanding west->east
    stage_macroblock(FileBlock("The nodes represent the cities,"), 2);
    gs->manager.transition(MICRO, "globe_opacity", ".2");
    gs->manager.set("label_size", "0");
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), "\\text{" + city + "}");
    }
    gs->manager.set("label_size", "{microblock_fraction} {microblock_fraction} 1 - *");
    gs->render_microblock();
    gs->manager.transition(MICRO, "label_size", "1");
    gs->render_microblock();

    stage_macroblock(FileBlock("and the edges connecting them are the rail lines."), netherlands_edges_1.size());
    for(pair<string, string> edge : netherlands_edges_1) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }

    double rotterdam_hash = HashableString("Rotterdam").get_hash();
    double groningen_hash = HashableString("Groningen").get_hash();

    vector<string> path = {"Rotterdam", "Utrecht", "Veenendaal", "Arnhem", "Zwolle", "Meppel", "Groningen"};
    stage_macroblock(FileBlock("To get from Rotterdam to Groningen,"), 3);
    gs->manager.transition(MACRO, "d", ".06");
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), "");
    }
    gs->render_microblock();
    // Label Rotterdam
    gs->config->transition_node_label(MICRO, rotterdam_hash, "\\text{Rotterdam}");
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);
    gs->render_microblock();
    // Label Groningen
    gs->config->transition_node_label(MICRO, groningen_hash, "\\text{Groningen}");
    gs->config->transition_node_color(MICRO, groningen_hash, 0xff00ff00);
    gs->render_microblock();

    stage_macroblock(FileBlock("this path seems obvious."), path.size());
    set_camera_to_lat_long(gs, netherlands_cities["Zwolle"], false, MACRO);
    gs->manager.transition(MACRO, "d", ".08");
    for(int i = 0; i < path.size() - 1; i++) {
        string city = path[i];
        string neighbor = path[i+1];
        gs->config->transition_edge_color(MICRO, HashableString(city).get_hash(), HashableString(neighbor).get_hash(), 0xffff0000);
        gs->render_microblock();
        gs->config->transition_node_color(MICRO, HashableString(neighbor).get_hash(), 0xffff0000);
    }
    gs->render_microblock();

    // Add more nodes
    stage_macroblock(FileBlock("And it is the shortest, but what about now?"), netherlands_edges_2.size() + 6);
    set_camera_to_lat_long(gs, center, false, MACRO);
    gs->manager.transition(MACRO, "d", ".055");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);

    for(pair<string, string> edge : netherlands_edges_2) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }

    gs->render_microblock();

    // Add even more nodes
    stage_macroblock(FileBlock("And now?"), netherlands_edges_3.size() + 4);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    for(pair<string, string> edge : netherlands_edges_3) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }
    gs->render_microblock();

    // Slide 9

    int per_bfs = 4;
    stage_macroblock(FileBlock("Here's a possible method:"), 1);
    uint32_t edge_dark = 0xff808080;
    gs->config->fade_all_node_colors(MICRO, edge_dark);
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->render_microblock();

    stage_macroblock(FileBlock("Rotterdam is our starting point, or our source."), 2);
    gs->manager.transition(MICRO, "d", ".045");
    set_camera_to_lat_long(gs, netherlands_cities["Rotterdam"], false, MICRO);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);
    gs->render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From Rotterdam, we explore its neighboring nodes for the goal, or the target."), FileBlock("If we don’t find it, we mark Rotterdam as explored and move on to its neighbors.")), per_bfs);
    unordered_set<double> border;
    unordered_set<double> visited;
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    int depth = 0;
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("We run the same check, and explore all their neighbors."), per_bfs + 4);
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("Still no target? Move on to the next set of nodes."), per_bfs + 4);
    set_camera_to_lat_long(gs, center, false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("At every level, check all the neighboring nodes for the target."), per_bfs);
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(SilenceBlock(2), per_bfs + 2);
    set_camera_to_lat_long(gs, netherlands_cities["Zwolle"], false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(SilenceBlock(1), 2);
    gs->render_microblock();
    reset_graph(gs, 0xff505050);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 6);
    trace_path(gs, {"Rotterdam", "'s-Hertogenbosch", "Arnhem", "Zwolle", "Meppel", "Groningen"}, 0xffffffff);

    stage_macroblock(FileBlock("This algorithm is known as breadth first search."), 1);
    reset_graph(gs, edge_dark);
    set_camera_to_lat_long(gs, center, false, MACRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("It will always find the shortest path because it checks all nodes at every level."), per_bfs * 5);
    gs->manager.transition(MACRO, "d", ".05");
    visited.clear();
    border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);
    reset_graph(gs, edge_dark);

    for(int i = 0; i < 5; i++) {
        bfs(g, gs, border, visited, i);
    }

    visited.clear();
    border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    stage_macroblock(FileBlock("If there was a path from Rotterdam to Groningen in four steps,"), 6);
    gs->render_microblock();
    reset_graph(gs, edge_dark);
    gs->render_microblock();
    gs->config->splash_node(rotterdam_hash);
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MICRO);
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    vec2 midpoint = ((netherlands_cities["Utrecht"] * 3) + netherlands_cities["Meppel"]) / 4.0f;
    set_camera_to_lat_long(gs, midpoint, false, MICRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("we would’ve found it on the fourth iteration, so five steps must be the shortest path."), per_bfs * 5 + 18);
    gs->manager.transition(MACRO, "d", ".06");
    for(int i = 0; i < 4; i++) {
        bfs(g, gs, border, visited, i);
    }
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, 4);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But there’s a problem. The algorithm thinks all these edges are the same length."), 5);
    gs->render_microblock();
    reset_graph(gs);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_all_edge_labels(MICRO, "1");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    gs->render_microblock();
}
