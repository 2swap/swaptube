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
    {"Tilburg", vec2(51.5555, 5.0913)},
    {"Meppel", vec2(52.7917, 6.1789)},
    {"Veenendaal", vec2(52.0231, 5.3889)},
    {"Almere", vec2(52.3508, 5.2647)},
};

vector<pair<string, string>> netherlands_edges_1 = {
    {"Rotterdam", "Utrecht"},
    {"Rotterdam", "Breda"},
    {"Breda", "Tilburg"},
    {"Tilburg", "Eindhoven"},
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
    {"Amsterdam", "The Hague"},
    {"Tilburg", "'s-Hertogenbosch"},
    {"Groningen", "Leeuwarden"},
    {"Almere", "Zwolle"},
    {"The Hague", "Utrecht"},
    {"Rotterdam", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Arnhem"},
};

vector<tuple<string, string, int>> netherlands_edge_lengths = {
    {"Rotterdam", "Utrecht", 3},
    {"Rotterdam", "Breda", 2},
    {"Breda", "Tilburg", 3},
    {"Tilburg", "Eindhoven", 1},
    {"Eindhoven", "'s-Hertogenbosch", 3},
    {"'s-Hertogenbosch", "Utrecht", 2},
    {"Arnhem", "Zwolle", 5},
    {"Zwolle", "Emmen", 2},
    {"Zwolle", "Meppel", 3},
    {"Meppel", "Groningen", 5},
    {"Meppel", "Leeuwarden", 2},
    {"Amsterdam", "The Hague", 5},
    {"Amsterdam", "Utrecht", 3},
    {"The Hague", "Rotterdam", 1},
    {"Tilburg", "'s-Hertogenbosch", 1},
    {"Groningen", "Leeuwarden", 2},
    {"The Hague", "Utrecht", 4},
    {"Rotterdam", "'s-Hertogenbosch", 4},
    {"'s-Hertogenbosch", "Arnhem", 2},
    {"Almere", "Zwolle", 4},
    {"Amsterdam", "Almere", 1},
    {"Veenendaal", "Arnhem", 2},
    {"Utrecht", "Veenendaal", 2},
};

void reset_graph(shared_ptr<GraphScene> gs, uint32_t col = 0xffffffff) {
    // Cleanup: transition all nodes and edges back to white and radius 1
    gs->config->fade_all_node_colors(MICRO, col);
    gs->config->fade_all_edge_colors(MICRO, col);
    gs->config->transition_all_node_radii(MICRO, 1);

    // Set rotterdam and groningen to red
    gs->config->fade_node_color(MICRO, HashableString("Rotterdam").get_hash(), 0xffff0000);
    gs->config->fade_node_color(MICRO, HashableString("Groningen").get_hash(), 0xffff0000);
}

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->label_color = 0xffffffff;
    gs->label_offset = vec2(0, 0.035);
    gs->edge_label_offset = 0.015;
    gs->label_size = vec2(0.6, 0.04);
    gs->manager.transition(MACRO, "globe_opacity", "1");
    vec2 center = (netherlands_cities["Utrecht"] + netherlands_cities["Zwolle"]) / 2.0;
    set_camera_to_lat_long(gs, center, true, MACRO);
    gs->manager.set("timey", "3"); // Hack
    gs->manager.set({
        {"midpoint_multiplier", "1.7"},
        {"points_radius_multiplier", "1.3"},
        {"node_labels_size", "1.1"},
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

    stage_macroblock(FileBlock("and the edges connecting them are the roads."), netherlands_edges_1.size());
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

    stage_macroblock(SilenceBlock(1), 1);
    uint32_t edge_dark = 0xff808080;
    gs->config->fade_all_node_colors(MICRO, edge_dark);
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), "");
    }
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    gs->render_microblock();

    vector<string> path = {"Rotterdam", "Utrecht", "Veenendaal", "Arnhem", "Zwolle", "Meppel", "Groningen"};
    stage_macroblock(FileBlock("The simplest way to find the shortest path from Rotterdam to Groningen looks like this:"), 12);
    gs->manager.set("node_labels_size", "1.4");
    gs->manager.transition(MACRO, "d", ".045");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    // Label Rotterdam
    gs->config->transition_node_label(MICRO, rotterdam_hash, "\\text{Rotterdam}");
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);
    gs->config->transition_node_radius(MICRO, rotterdam_hash, 1.5);
    gs->render_microblock();
    set_camera_to_lat_long(gs, center, false, MICRO);
    gs->render_microblock();
    // Label Groningen
    gs->config->transition_node_label(MICRO, groningen_hash, "\\text{Groningen}");
    gs->config->transition_node_color(MICRO, groningen_hash, 0xffff0000);
    gs->config->transition_node_radius(MICRO, groningen_hash, 1.5);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    // Slide 9

    int per_bfs = 4;
    reset_graph(gs, edge_dark);

    stage_macroblock(FileBlock("Starting from Rotterdam,"), 3);
    gs->manager.transition(MACRO, "d", ".025");
    set_camera_to_lat_long(gs, netherlands_cities["Rotterdam"], false, MACRO);
    gs->render_microblock();
    gs->config->transition_node_radius(MICRO, rotterdam_hash, 2);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("we’re first going to explore all its neighbors, looking for Groningen."), 4);
    unordered_set<double> border;
    unordered_set<double> visited;
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    int depth = 0;
    bfs(g, gs, border, visited, depth, 1);
    gs->render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(0.5), FileBlock("Since Groningen isn’t one of these neighbouring nodes, we’ll mark Rotterdam as explored and keep searching.")), 7);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++, 2);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("So now, we’ll branch out from these four nodes to explore all their neighbors."), per_bfs + 4);
    gs->manager.transition(MACRO, "d", ".035");
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(FileBlock("And we keep branching out this way, exploring all neighbouring nodes until we stumble upon Groningen."), per_bfs * 3);
    set_camera_to_lat_long(gs, netherlands_cities["Meppel"], false, MACRO);
    bfs(g, gs, border, visited, depth++);
    bfs(g, gs, border, visited, depth++);
    bfs(g, gs, border, visited, depth++);

    stage_macroblock(SilenceBlock(.5), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 2);
    set_camera_to_lat_long(gs, center, false, MACRO);
    gs->manager.transition(MACRO, "d", ".04");
    gs->render_microblock();
    reset_graph(gs, 0xff505050);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(2), 8);
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "'s-Hertogenbosch", "Arnhem", "Zwolle", "Meppel", "Groningen"}, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("This algorithm is known as breadth first search."), 1);
    reset_graph(gs, edge_dark);
    gs->render_microblock();

    stage_macroblock(FileBlock("It will always find the shortest path because it checks all nodes"), 1);
    gs->manager.transition(MACRO, "d", ".035");
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    visited.clear();
    border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);
    reset_graph(gs, edge_dark);
    gs->render_microblock();

    stage_macroblock(FileBlock("1 step away,"), per_bfs + 4);
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, 0);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("then 2 steps away,"), per_bfs + 5);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, 1);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("and so on."), SilenceBlock(.8)), 3 * per_bfs + 2);
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, visited, 2);
    bfs(g, gs, border, visited, 3);
    bfs(g, gs, border, visited, 4);

    visited.clear();
    border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    stage_macroblock(FileBlock("So if there was a path from Rotterdam to Groningen in four steps,"), 9);
    gs->render_microblock();
    reset_graph(gs, edge_dark);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(rotterdam_hash);
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_node_radius(MICRO, groningen_hash, 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("we would’ve found it on the fourth iteration."), per_bfs * 4);
    gs->manager.transition(MACRO, "d", ".03");
    set_camera_to_lat_long(gs, netherlands_cities["Meppel"], false, MACRO);
    for(int i = 0; i < 4; i++) {
        bfs(g, gs, border, visited, i);
    }

    stage_macroblock(FileBlock("But we didn't. So the shortest path must be five steps."), 9 + per_bfs);
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

    stage_macroblock(SilenceBlock(1), 1);
    set_camera_to_lat_long(gs, center, false, MACRO);
    gs->manager.transition(MACRO, "d", ".045");
    gs->render_microblock();

    stage_macroblock(FileBlock("But there’s a problem. The algorithm thinks all these edges are the same length."), 5);
    gs->render_microblock();
    reset_graph(gs, edge_dark);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_all_edge_labels(MICRO, "1");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("That’s not true."), SilenceBlock(1)), 2);
    gs->render_microblock();
    for(auto& [city1, city2, length] : netherlands_edge_lengths) {
        double hash1 = HashableString(city1).get_hash();
        double hash2 = HashableString(city2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, "");
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("So let's add weights to represent distances between these nodes."), 2);
    gs->manager.transition(MACRO, "d", ".045");
    // Add edge labels
    for(auto& [city1, city2, length] : netherlands_edge_lengths) {
        double hash1 = HashableString(city1).get_hash();
        double hash2 = HashableString(city2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(length));
    }
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Now the shortest path is much harder to figure out."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So Dijkstra needed a better method."), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    gs->render_microblock();
}
