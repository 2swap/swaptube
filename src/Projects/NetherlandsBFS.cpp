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
    //{"Maastricht", vec2(50.8514, 5.6900)},
    {"Arnhem", vec2(51.9851, 5.8987)},
    {"Zwolle", vec2(52.5168, 6.0830)},
    {"Emmen", vec2(52.7795, 6.9061)},
    {"Groningen", vec2(53.2194, 6.5665)},
    {"Leeuwarden", vec2(53.2012, 5.7999)},
    {"'s-Hertogenbosch", vec2(51.6978, 5.3037)},
    {"Tillburg", vec2(51.5555, 5.0913)},
    {"Meppel", vec2(52.7917, 6.1789)},
};

vector<pair<string, string>> netherlands_edges_1 = {
    {"Rotterdam", "Utrecht"},
    {"Rotterdam", "Breda"},
    {"Breda", "Tillburg"},
    {"Tillburg", "Eindhoven"},
    //{"Eindhoven", "Maastricht"},
    {"Eindhoven", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Utrecht"},
    {"Utrecht", "Arnhem"},
    {"Arnhem", "Zwolle"},
    {"Zwolle", "Emmen"},
    {"Zwolle", "Meppel"},
    {"Meppel", "Groningen"},
    {"Meppel", "Leeuwarden"},
    {"The Hague", "Rotterdam"},
    {"Amsterdam", "Utrecht"},
};

vector<pair<string, string>> netherlands_edges_2 = {
    {"Amsterdam", "The Hague"},
    {"Amsterdam", "Zwolle"},
    {"Tillburg", "'s-Hertogenbosch"},
};

vector<pair<string, string>> netherlands_edges_3 = {
    {"Groningen", "Leeuwarden"},
    {"The Hague", "Utrecht"},
    {"Rotterdam", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Arnhem"},
};

vector<uint32_t> colors_by_depth = {
    0xffff0000, // red
    0xffffa500, // orange
    0xffffff00, // yellow
    0xff00ff00, // green
    0xff00ffff, // cyan
    0xff0000ff, // blue
    0xff800080, // purple
    0xffff00ff, // magenta
    0xffff0080, // pink
};

void bfs(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double>& border, unordered_set<double>& visited, int depth) {
    // 4 microblocks / visual transformations

    // 1: splash and grow all nodes currently in the border
    for(double node : border) {
        gs->config->fade_node_color(MICRO, node, colors_by_depth[depth == 0 ? 0 : depth - 1]);
        gs->config->transition_node_radius(MICRO, node, 2);
    }
    gs->render_microblock();

    // 2: color all outgoing edges
    unordered_set<double> next_border;
    for(double node : border) {
        unordered_set<double> neighbors = g->get_neighbors(node);
        for(double neighbor : neighbors) {
            if(visited.find(neighbor) != visited.end()) continue;
            next_border.insert(neighbor);
            gs->config->transition_edge_color(MICRO, node, neighbor, colors_by_depth[depth]);
        }
    }
    gs->render_microblock();

    // 3: Color all neighboring nodes, which will become the next border
    for(double node : next_border) {
        gs->config->transition_node_color(MICRO, node, colors_by_depth[depth]);
    }
    gs->render_microblock();

    // 4: Shrink the old border and grow the new border
    for(double node : border) {
        gs->config->transition_node_radius(MICRO, node, 0);
    }
    for(double node : next_border) {
        gs->config->transition_node_radius(MICRO, node, 2);
    }
    gs->render_microblock();

    // Update visited and border sets
    visited.insert(next_border.begin(), next_border.end());
    border = next_border;
}

void reset_graph(shared_ptr<GraphScene> gs) {
    // Cleanup: transition all nodes and edges back to white and radius 1
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->transition_all_node_radii(MICRO, 1);
}

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->label_color = 0xffffffff;
    gs->label_offset = vec2(0, 0.02);
    gs->label_size = vec2(0.4, 0.04);
    gs->manager.transition(MACRO, "globe_opacity", "1");
    set_camera_to_lat_long(gs, vec2(52.5, 5.5), true, MACRO);
    gs->manager.set({
        {"physics_multiplier","0"},
        {"d", ".07"},
    });

    // Load Netherlands map
    stage_macroblock(FileBlock("Here's a simplified graph of the Netherlands."), netherlands_cities.size());

    // Plot cities as nodes and roads as edges, expanding east->west
    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        double hash = HashableString(city).get_hash();
        g->add_node(new HashableString(city));
        g->move_node(hash, position);
        gs->render_microblock();
    }

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

    vector<string> path = {"Rotterdam", "Utrecht", "Arnhem", "Zwolle", "Meppel", "Groningen"};
    stage_macroblock(FileBlock("To get from Rotterdam to Groningen,"), 3);
    set_camera_to_lat_long(gs, netherlands_cities["Rotterdam"], false, MACRO);
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
    set_camera_to_lat_long(gs, netherlands_cities["Groningen"], false, MACRO);
    gs->manager.transition(MACRO, "d", ".10");
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
    set_camera_to_lat_long(gs, vec2(52.5, 5.5), false, MACRO);
    gs->manager.transition(MACRO, "d", ".07");
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
    gs->render_microblock();
    gs->render_microblock();

    // Add even more nodes
    stage_macroblock(FileBlock("And now?"), netherlands_edges_3.size() + 4);
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
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("It's not so obvious anymore."), 1);
    gs->render_microblock();

    // Slide 9

    int per_bfs = 4;
    stage_macroblock(FileBlock("Here's a possible method:"), 1);
    uint32_t edge_dark = 0xff808080;
    gs->config->fade_all_node_colors(MICRO, edge_dark);
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->render_microblock();

    stage_macroblock(FileBlock("Starting from the source, we explore its neighboring nodes for the target."), 1 + per_bfs);
    set_camera_to_lat_long(gs, netherlands_cities["Rotterdam"], false, MICRO);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);

    unordered_set<double> border;
    unordered_set<double> next_border;
    unordered_set<double> visited;
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    int depth = 0;
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("If it's not there, we move on to those explored nodes, checking _their_ neighbors."), per_bfs + 4);
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("If we don't find it, we expand again."), per_bfs + 4);
    set_camera_to_lat_long(gs, netherlands_cities["Meppel"], false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("And we keep expanding out until we reach the target."), per_bfs * 2);
    set_camera_to_lat_long(gs, netherlands_cities["Groningen"], false, MACRO);
    bfs(g, gs, border, visited, depth++);
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(SilenceBlock(1), 1);
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("This algorithm is known as breadth first search."), 1);
    set_camera_to_lat_long(gs, netherlands_cities["Arnhem"], false, MACRO);
    gs->render_microblock();
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->config->fade_all_node_colors(MICRO, edge_dark);

    stage_macroblock(FileBlock("It will always find the shortest path because it checks all nodes at every level."), per_bfs * 5);
    gs->manager.transition(MACRO, "d", ".065");
    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);
    reset_graph(gs);

    for(int i = 0; i < 5; i++) {
        bfs(g, gs, border, visited, i);
    }

    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    stage_macroblock(FileBlock("If there was a path from Rotterdam to Groningen in five steps,"), 6);
    gs->render_microblock();
    reset_graph(gs);
    gs->render_microblock();
    gs->config->splash_node(rotterdam_hash);
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MICRO);
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    vec2 midpoint = (netherlands_cities["Utrecht"] * 3 + netherlands_cities["Meppel"]) / 4.0f;
    set_camera_to_lat_long(gs, midpoint, false, MICRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("we would’ve found it on the fifth iteration, so six steps must be the shortest path."), per_bfs * 5 + 18);
    gs->manager.transition(MACRO, "d", ".07");
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

    stage_macroblock(FileBlock("But there’s a problem. The algorithm thinks all these edges are the same length."), 5);
    gs->render_microblock();
    reset_graph(gs);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_all_edge_labels(MICRO, "\\small{1}");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    gs->render_microblock();
}
