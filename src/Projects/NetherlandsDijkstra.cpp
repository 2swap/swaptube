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

unordered_map<string, int> true_city_costs = {
    {"Rotterdam", 0},
    {"The Hague", 1},
    {"Amsterdam", 6},
    {"Breda", 2},
    {"Tillburg", 5},
    {"Eindhoven", 6},
    {"'s-Hertogenbosch", 4},
    {"Arnhem", 6},
    {"Zwolle", 11},
    {"Emmen", 13},
    {"Groningen", 18},
    {"Leeuwarden", 16},
    {"Meppel", 14},
    {"Utrecht", 3},
    {"Veenendaal", 5},
    {"Almere", 7},
};

vector<tuple<string, string, int>> netherlands_edges = {
    {"Rotterdam", "Utrecht", 3},
    {"Rotterdam", "Breda", 2},
    {"Breda", "Tillburg", 3},
    {"Tillburg", "Eindhoven", 1},
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
    {"Tillburg", "'s-Hertogenbosch", 1},
    {"Groningen", "Leeuwarden", 2},
    {"The Hague", "Utrecht", 4},
    {"Rotterdam", "'s-Hertogenbosch", 4},
    {"'s-Hertogenbosch", "Arnhem", 2},
    {"Almere", "Zwolle", 4},
    {"Amsterdam", "Almere", 1},
    {"Veenendaal", "Arnhem", 2},
    {"Utrecht", "Veenendaal", 2},
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

void attempt_relax_edge(shared_ptr<GraphScene> gs, double from, double to, unordered_map<double, int>& costs) {
    // Find the edge weight
    int weight = 0;
    for(auto& [node, neighbor, w] : netherlands_edges) {
        double hash1 = HashableString(node).get_hash();
        double hash2 = HashableString(neighbor).get_hash();
        if((hash1 == from && hash2 == to) || (hash1 == to && hash2 == from)) {
            weight = w;
            break;
        }
    }

    int tentative_cost = costs[from] + weight;

    gs->config->transition_edge_color(MICRO, from, to, 0xffff0000);
    bool updated = false;
    if(tentative_cost < costs[to]) {
        costs[to] = tentative_cost;
        updated = true;
    }
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, to, 0xffff0000);
    if(updated) gs->config->transition_node_label(MICRO, to, to_string(tentative_cost));
    gs->render_microblock();

    gs->config->fade_edge_color(MICRO, to, from, 0xffffffff);
    gs->config->fade_node_color(MICRO, to, 0xffffffff);
    gs->render_microblock();
}

void put_in_visited_set(shared_ptr<GraphScene> gs, double node_hash, unordered_set<double>& visited) {
    gs->config->transition_node_color(MICRO, node_hash, 0xffffffff);
    gs->config->transition_node_radius(MICRO, node_hash, .7);
    gs->config->transition_node_label(MICRO, node_hash, "");
    visited.insert(node_hash);
    gs->render_microblock();
}

void attempt_relax_all_edges(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double from, unordered_map<double, int>& costs, unordered_set<double>& visited) {
    if(!g->get_neighbors(from).empty()) {
        unordered_map<double, bool> updated_neighbors;
        for(double to : g->get_neighbors(from)) {
            if(visited.find(to) != visited.end()) continue;
            // Find the edge weight
            int weight = 0;
            for(auto& [node, neighbor, w] : netherlands_edges) {
                double hash1 = HashableString(node).get_hash();
                double hash2 = HashableString(neighbor).get_hash();
                if((hash1 == from && hash2 == to) || (hash1 == to && hash2 == from)) {
                    weight = w;
                    break;
                }
            }

            int tentative_cost = costs[from] + weight;

            gs->config->transition_edge_color(MICRO, from, to, 0xffff0000);
            if(tentative_cost < costs[to]) {
                costs[to] = tentative_cost;
                updated_neighbors[to] = true;
            }
        }

        gs->render_microblock();

        for(double to : g->get_neighbors(from)) {
            if(updated_neighbors.find(to) == updated_neighbors.end()) continue;
            gs->config->transition_node_color(MICRO, to, 0xffff0000);
            gs->config->transition_node_label(MICRO, to, to_string(costs[to]));
        }

        gs->render_microblock();

        for(double to : g->get_neighbors(from)) {
            gs->config->fade_edge_color(MICRO, to, from, 0xffffffff);
            gs->config->fade_node_color(MICRO, to, 0xffffffff);
        }

        gs->render_microblock();
    }

    put_in_visited_set(gs, from, visited);
}

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->label_color = 0xff000000;
    gs->label_size = vec2(0.55, 0.055);
    gs->manager.transition(MACRO, "globe_opacity", "0.3");
    vec2 center = (netherlands_cities["Utrecht"] + netherlands_cities["Zwolle"]) / 2.0;
    set_camera_to_lat_long(gs, center, true, MACRO);
    gs->manager.set({
        {"physics_multiplier","0"},
        {"d", ".05"},
    });

    double groningen_hash = HashableString("Groningen").get_hash();
    double rotterdam_hash = HashableString("Rotterdam").get_hash();
    // Plot cities as nodes and roads as edges, expanding east->west

    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        double hash = HashableString(city).get_hash();
        g->add_node(new HashableString(city));
        g->move_node(hash, position);
    }

    for (auto& [city1, city2, dist] : netherlands_edges) {
        double hash1 = HashableString(city1).get_hash();
        double hash2 = HashableString(city2).get_hash();
        g->add_edge(hash1, hash2);
    }

    stage_macroblock(FileBlock("So let's add distances."), 1);
    for (auto& [city1, city2, dist] : netherlands_edges) {
        double hash1 = HashableString(city1).get_hash();
        double hash2 = HashableString(city2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(dist));
    }
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 2);
    gs->manager.transition(MICRO, "edge_weights_size", "1");
    gs->render_microblock();
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, "");
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("Given these edge weights, the shortest path is much harder to figure out."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra realized that it wasn’t the number of edges to the source that mattered,"), 11);
    gs->config->fade_all_edge_colors(MICRO, 0xff505050);
    gs->config->fade_all_node_colors(MICRO, 0xff505050);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->set_edge_color(rotterdam_hash, HashableString("'s-Hertogenbosch").get_hash(), 0xffffffff);
    gs->render_microblock();
    gs->config->set_edge_color(HashableString("'s-Hertogenbosch").get_hash(), HashableString("Arnhem").get_hash(), 0xffffffff);
    gs->render_microblock();
    gs->config->set_edge_color(HashableString("Arnhem").get_hash(), HashableString("Zwolle").get_hash(), 0xffffffff);
    gs->render_microblock();
    gs->config->set_edge_color(HashableString("Zwolle").get_hash(), HashableString("Meppel").get_hash(), 0xffffffff);
    gs->render_microblock();
    gs->config->set_edge_color(HashableString("Meppel").get_hash(), groningen_hash, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("it was the distance to the source."), 1);
    // Render wavefront expanding from source
    gs->render_microblock();

    stage_macroblock(FileBlock("So instead, he could explore the closest node to the source, and then the next closest node, and so on."), 3);
    gs->render_microblock();
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("Breda").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("He’d track the shortest known distance to the node, called its cost."), 1);
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, to_string(true_city_costs[city]));
        gs->config->transition_node_radius(MICRO, hash, 2.5);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("That way, he could always explore nodes from lowest to highest cost, or closest to furthest from the source."), 6 + netherlands_cities.size());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    // in order of increasing cost
    unordered_map<string, int> removable_cities = true_city_costs;
    while(!removable_cities.empty()) {
        int min_city_cost = 1000;
        string min_city_name = "";
        double min_city = 0;
        for(auto& [city, coords] : removable_cities) {
            double hash = HashableString(city).get_hash();
            if(true_city_costs[city] < min_city_cost) {
                min_city_cost = true_city_costs[city];
                min_city_name = city;
                min_city = hash;
            }
        }
        removable_cities.erase(min_city_name);
        gs->config->splash_node(min_city);
        gs->render_microblock();
    }
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("But until he actually explored the graph, he didn’t know the shortest distance to any node yet."), 5);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, "");
        gs->config->transition_node_radius(MICRO, hash, 1);
    }
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("So at the beginning, the source's cost was zero,"), 4);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_node_radius(MICRO, rotterdam_hash, 2.5);
    gs->config->transition_node_label(MICRO, rotterdam_hash, "0");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("and every other node was infinity."), 4);
    gs->render_microblock();
    gs->render_microblock();
    for(auto& [city, coords] : netherlands_cities) {
        if(city == "Rotterdam") continue;
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_radius(MICRO, hash, 2.5);
    }
    for(auto& [city, coords] : netherlands_cities) {
        if(city == "Rotterdam") continue;
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, "\\infty");
    }
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Then, starting from the source, Dijkstra explored each of the neighboring nodes."), 2);
    int step = 0;
    for(auto& node : g->get_neighbors(rotterdam_hash)) {
        gs->config->transition_edge_color(MICRO, rotterdam_hash, node, 0xfffffffe);
    }
    gs->render_microblock();

    for(auto& node : g->get_neighbors(rotterdam_hash)) {
        gs->config->transition_node_color(MICRO, node, 0xfffffffe);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("At every node, he'd ask: is the current path the shortest path so far?"), 3);
    // Zoom in on the edge
    gs->manager.transition(MICRO, "d", "0.02");
    set_camera_to_lat_long(gs, (netherlands_cities["Rotterdam"] + netherlands_cities["The Hague"]) / 2, false, MICRO);
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, rotterdam_hash, HashableString("The Hague").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->fade_edge_color(MICRO, HashableString("The Hague").get_hash(), rotterdam_hash, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, this edge only has a weight of 1 while the node’s current cost is infinity."), 4);
    // Zoom out and recenter
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, rotterdam_hash, HashableString("The Hague").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("This path is the shortest yet to that node [1 < inf], so Dijkstra updated its cost to 1."), 3);
    unordered_map<double, int> costs;
    unordered_set<double> visited;
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        costs[hash] = 1000;
    }
    costs[rotterdam_hash] = 0;
    attempt_relax_edge(gs, rotterdam_hash, HashableString("The Hague").get_hash(), costs);

    stage_macroblock(FileBlock("After checking all of Rotterdam’s edges and updating the neighbors’ costs,"), 9);
    gs->manager.transition(MACRO, "d", "0.05");
    set_camera_to_lat_long(gs, center, false, MACRO);
    attempt_relax_edge(gs, rotterdam_hash, HashableString("Utrecht").get_hash(), costs);
    attempt_relax_edge(gs, rotterdam_hash, HashableString("Breda").get_hash(), costs);
    attempt_relax_edge(gs, rotterdam_hash, HashableString("'s-Hertogenbosch").get_hash(), costs);

    stage_macroblock(FileBlock("Dijkstra marked it as explored. If this was breadth first search,"), 2);
    put_in_visited_set(gs, rotterdam_hash, visited);
    gs->render_microblock();

    stage_macroblock(FileBlock("he could pick any neighbor to explore next."), 2);
    gs->render_microblock();
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->config->splash_node(HashableString("Utrecht").get_hash());
    gs->config->splash_node(HashableString("'s-Hertogenbosch").get_hash());
    gs->config->splash_node(HashableString("Breda").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("But he wanted to prioritize closer nodes."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So he explored the nodes from lowest to highest cost."), 4);
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("Breda").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("Utrecht").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("'s-Hertogenbosch").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("With a cost of 1, this node was next."), 2);
    gs->render_microblock();
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Like before, he checked each of its edges to reduce the neighbors’ costs."), 2);
    double hague_hash = HashableString("The Hague").get_hash();
    for(auto& node : g->get_neighbors(hague_hash)) {
        if(node == rotterdam_hash) continue;
        gs->config->transition_edge_color(MICRO, hague_hash, node, 0xfffffffe);
    }
    gs->render_microblock();

    for(auto& node : g->get_neighbors(hague_hash)) {
        if(node == rotterdam_hash) continue;
        gs->config->transition_node_color(MICRO, node, 0xfffffffe);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("While Dijkstra could update this neighbor,"), 3);
    attempt_relax_edge(gs, hague_hash, HashableString("Amsterdam").get_hash(), costs);

    stage_macroblock(FileBlock("he couldn’t update this one."), 3);
    double utrecht_hash = HashableString("Utrecht").get_hash();
    attempt_relax_edge(gs, hague_hash, utrecht_hash, costs);

    stage_macroblock(FileBlock("This path has a cost of 5,"), 4);
    trace_path(gs, {"Rotterdam", "The Hague", "Utrecht"}, 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("but the node’s cost was 3."), 2);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->config->splash_node(utrecht_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("There was already a shorter path, so no need to update it."), 3);
    trace_path(gs, {"Rotterdam", "Utrecht"}, 0xffff0000);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("This continued for the rest of the graph."), SilenceBlock(4)), 17);
    put_in_visited_set(gs, hague_hash, visited);
    gs->render_microblock();
    attempt_relax_all_edges(g, gs, HashableString("Breda").get_hash(), costs, visited);
    gs->render_microblock();
    attempt_relax_all_edges(g, gs, HashableString("Utrecht").get_hash(), costs, visited);
    gs->render_microblock();
    attempt_relax_all_edges(g, gs, HashableString("'s-Hertogenbosch").get_hash(), costs, visited);
    gs->render_microblock();

    stage_macroblock(FileBlock("Sometimes he had to update a node's costs a few times."), 1);
    gs->manager.transition(MACRO, "d", "0.02");
    set_camera_to_lat_long(gs, netherlands_cities["'s-Hertogenbosch"], false, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, he first reached this node through this path with a cost of 7."), 6);
    gs->config->splash_node(HashableString("Eindhoven").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "'s-Hertogenbosch", "Eindhoven"}, 0xffff0000);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);

    stage_macroblock(FileBlock("But later, he found a shorter path of cost 6 and updated the node’s cost again."), 8);
    gs->manager.transition(MACRO, "d", "0.04");
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MICRO);
    trace_path(gs, {"Rotterdam", "Breda", "Tillburg", "Eindhoven"}, 0xffff0000);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    attempt_relax_all_edges(g, gs, HashableString("Tillburg").get_hash(), costs, visited);

    stage_macroblock(SilenceBlock(5), 28);
    gs->manager.transition(MACRO, "d", "0.02");
    set_camera_to_lat_long(gs, netherlands_cities["Meppel"], false, MACRO);
    while(true){
        double closest = -1;
        double closest_cost = 100;
        for(auto& [city, coords] : netherlands_cities) {
            if(visited.find(HashableString(city).get_hash()) == visited.end()) {
                if(costs[HashableString(city).get_hash()] < closest_cost) {
                    closest_cost = costs[HashableString(city).get_hash()];
                    closest = HashableString(city).get_hash();
                }
            }
        }
        if(closest == HashableString("Meppel").get_hash()) break;
        if(closest == -1) break;
        attempt_relax_all_edges(g, gs, closest, costs, visited);
    }

    stage_macroblock(FileBlock("The first time Dijkstra reached Groningen"), 4);
    attempt_relax_all_edges(g, gs, HashableString("Meppel").get_hash(), costs, visited);

    stage_macroblock(FileBlock("it had a cost of 19."), 2);
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("But it didn’t have the lowest cost out of all the unexplored nodes."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("There could still be a shorter path, just like earlier."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("He continued exploring nodes in cost order, and found this path with cost 18."), 14);
    gs->manager.transition(MACRO, "d", "0.05");
    set_camera_to_lat_long(gs, center, false, MACRO);
    attempt_relax_edge(gs, HashableString("Leeuwarden").get_hash(), groningen_hash, costs);
    put_in_visited_set(gs, HashableString("Leeuwarden").get_hash(), visited);
    trace_path(gs, {"Rotterdam", "The Hague", "Amsterdam", "Almere", "Zwolle", "Meppel", "Leeuwarden", "Groningen"}, 0xffff0000);
    gs->render_microblock();
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Now out of all the unexplored nodes, Groningen had the lowest cost."), 1);
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra was confident he had found the shortest path."), 1);
    gs->render_microblock();
    return;

    stage_macroblock(FileBlock("That’s because on a road network, the cost always increases along a path."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So if the lowest cost among the unexplored nodes was 18,"), 1);
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra must have explored all the paths with costs under 18."), 1);
    gs->config->transition_edge_color(MICRO, HashableString("Leeuwarden").get_hash(), groningen_hash, 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("Meppel").get_hash(), groningen_hash, 0x00000000);
    gs->render_microblock();

    stage_macroblock(FileBlock("And none of those paths reached Groningen, otherwise he would’ve updated its cost earlier."), 3);
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("It’s just like the levels of breadth-first search."), 1);

    gs->render_microblock();
    stage_macroblock(FileBlock("By exploring from lowest to highest cost,"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("Dijkstra’s algorithm guaranteed the shortest path to any target."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("To get directions for the shortest path,"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("we just need to keep track of what nodes are able to relax others — a sort of predecessor list."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Then at the end, we can easily work backwards to get the nodes in order."), 10);
    // Add all true costs to the graph
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, to_string(true_city_costs[city]));
        gs->config->transition_node_radius(MICRO, hash, 2.5);
    }
    gs->render_microblock();
    trace_path(gs, {"Groningen", "Leeuwarden", "Meppel", "Zwolle", "Almere", "Amsterdam", "The Hague", "Rotterdam"}, 0xffff0000);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();
}
