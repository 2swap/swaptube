#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

uint32_t dark_blue = 0xff505070;

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

unordered_map<string, int> true_city_costs = {
    {"Rotterdam", 0},
    {"The Hague", 1},
    {"Amsterdam", 6},
    {"Breda", 2},
    {"Tilburg", 5},
    {"Eindhoven", 6},
    {"'s-Hertogenbosch", 4},
    {"Arnhem", 7},
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
    {"'s-Hertogenbosch", "Arnhem", 3},
    {"Almere", "Zwolle", 4},
    {"Amsterdam", "Almere", 1},
    {"Veenendaal", "Arnhem", 2},
    {"Utrecht", "Veenendaal", 2},
};

int infinity_color = 0xffc08080;
void restore_graph_colors(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double> visited, unordered_map<double, int> costs) {
    gs->config->fade_all_edge_colors(MICRO, infinity_color);
    gs->config->fade_all_node_colors(MICRO, infinity_color);
    // nodes with cost of infinity should be gray
    for(auto& [hash, cost] : costs) {
        if (cost >= 100) {
            gs->config->fade_node_color(MICRO, hash, infinity_color);
        }
        if (cost < 100) {
            gs->config->fade_node_color(MICRO, hash, 0xffffffff);
            for(double neighbor : g->get_neighbors(hash)) {
                gs->config->fade_edge_color(MICRO, hash, neighbor, 0xffffffff);
            }
        }
    }
    for(double node : visited) {
        gs->config->fade_node_color(MICRO, node, visited_color);
        // For each neighbor, fade edge color to visited color
        for(double neighbor : g->get_neighbors(node)) {
            gs->config->fade_edge_color(MICRO, node, neighbor, visited_color);
        }
    }
    double groningen_hash = HashableString("Groningen").get_hash();
    gs->config->fade_node_color(MICRO, groningen_hash, 0xffff0000);
}

void transition_all_edges_from_node(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double node_hash, uint32_t color) {
    for(double neighbor : g->get_neighbors(node_hash)) {
        gs->config->transition_edge_color(MICRO, node_hash, neighbor, color);
    }
    gs->config->transition_node_color(MICRO, node_hash, color);
    gs->render_microblock();
    for(double neighbor : g->get_neighbors(node_hash)) {
        gs->config->transition_node_color(MICRO, neighbor, color);
    }
    gs->render_microblock();
}

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
    gs->config->transition_node_radius(MICRO, to, 3);
    if(updated) gs->config->transition_node_label(MICRO, to, to_string(tentative_cost));
    gs->render_microblock();

    gs->config->fade_edge_color(MICRO, to, from, visited_color);
    gs->config->fade_node_color(MICRO, to, 0xffffffff);
    gs->render_microblock();
}

void put_in_visited_set(shared_ptr<GraphScene> gs, double node_hash, unordered_set<double>& visited) {
    gs->config->fade_node_color(MICRO, node_hash, visited_color);
    gs->config->transition_node_radius(MICRO, node_hash, 2);
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
            if(visited.find(to) != visited.end()) continue;
            gs->config->transition_node_color(MICRO, to, 0xffff0000);
            if(updated_neighbors.find(to) == updated_neighbors.end()) continue;
            gs->config->transition_node_radius(MICRO, to, 3);
            gs->config->transition_node_label(MICRO, to, to_string(costs[to]));
        }

        gs->render_microblock();

        for(double to : g->get_neighbors(from)) {
            if(visited.find(to) != visited.end()) continue;
            gs->config->fade_edge_color(MICRO, to, from, visited_color);
            gs->config->fade_node_color(MICRO, to, 0xffffffff);
        }

        gs->render_microblock();
    }

    put_in_visited_set(gs, from, visited);
}

void color_all_darkblue_except_endpoints(shared_ptr<GraphScene> gs, shared_ptr<Graph> g, double source, double target) {
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        if(hash != source && hash != target) {
            gs->config->fade_node_color(MICRO, hash, dark_blue);
        }
        for(double neighbor : g->get_neighbors(hash)) {
            if(neighbor != source && neighbor != target) {
                gs->config->fade_edge_color(MICRO, hash, neighbor, dark_blue);
            }
        }
    }
    gs->config->fade_node_color(MICRO, source, 0xffff0000);
    gs->config->fade_node_color(MICRO, target, 0xffff0000);
}

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->label_color = 0xff000000;
    gs->label_size = vec2(0.55, 0.055);
    gs->edge_label_offset = 0.017;
    gs->manager.set("globe_opacity", "0.3");
    vec2 center = (netherlands_cities["Utrecht"] + netherlands_cities["Zwolle"]) / 2.0;
    set_camera_to_lat_long(gs, center, true, MACRO);
    gs->manager.set({
        {"physics_multiplier","0"},
        {"d", ".05"},
        {"midpoint_multiplier", "1.7"},
        {"edge_weights_size", "1"},
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
        gs->config->add_edge_if_missing(hash1, hash2);
        gs->config->set_edge_color(hash1, hash2, 0xffffffff);
        g->add_edge(hash1, hash2);
    }
    gs->config->set_node_color(rotterdam_hash, 0xffff0000);
    gs->config->set_node_color(groningen_hash, 0xffff0000);

    for (auto& [city1, city2, dist] : netherlands_edges) {
        double hash1 = HashableString(city1).get_hash();
        double hash2 = HashableString(city2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(dist));
    }
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    gs->config->transition_all_node_radii(MICRO, 1);

    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, "");
    }

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("His idea was this."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("First, he wanted to keep track of the shortest distance from the source to each node."), 1);
    gs->render_microblock();


    unordered_map<string, int> removable_cities = true_city_costs;
    vector<string> cities_in_order;
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
        cities_in_order.push_back(min_city_name);
    }
    stage_macroblock(FileBlock("This is also known as its cost."), 3);
    gs->render_microblock();
    gs->render_microblock();
    int index = 0;
    for(const string& city : cities_in_order) {
        double hash = HashableString(city).get_hash();
        char index_char = 'a' + index;
        index++;
        gs->config->transition_node_label(MICRO, hash, "C_{" + string(1, index_char) + "}");
        gs->config->transition_node_radius(MICRO, hash, 2.5);
    }
    gs->render_microblock();

    double leeuwarden_hash = HashableString("Leeuwarden").get_hash();
    stage_macroblock(FileBlock("His goal? If he could find the shortest path from the source to one node,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("then he could build up other shortest paths starting from this new node."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("And then the next. And on and on."), 13);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "The Hague"}, 0xffff0000);
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    splash_edge_path(MICRO, gs, {"Rotterdam", "The Hague"});
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    splash_edge_path(MICRO, gs, {"Rotterdam", "Breda"});
    trace_path(gs, {"Rotterdam", "Breda"}, 0xffff0000);
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("The cost of the starting node was 0,"), 2);
    gs->render_microblock();
    gs->config->transition_node_radius(MICRO, rotterdam_hash, 3);
    gs->config->transition_node_label(MICRO, rotterdam_hash, "0");
    gs->render_microblock();

    stage_macroblock(FileBlock("but since he hasn’t actually explored the rest of graph yet,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("he set all the other nodes to infinity."), 8);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
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
        gs->config->fade_node_color(MICRO, hash, infinity_color);
    }
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Then, just like breadth-first search,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra started from the source and explored"), 1);
    // Zoom in on the source node
    gs->manager.transition(MICRO, "d", "0.035");
    set_camera_to_lat_long(gs, netherlands_cities["Rotterdam"], false, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("each of its neighboring nodes."), 3);
    set_camera_to_lat_long(gs, (netherlands_cities["Rotterdam"] + netherlands_cities["The Hague"]) / 2, false, MICRO);
    gs->render_microblock();
    int step = 0;
    for(auto& node : g->get_neighbors(rotterdam_hash)) {
        gs->config->transition_edge_color(MICRO, rotterdam_hash, node, 0xfffffffe);
    }
    gs->render_microblock();

    for(auto& node : g->get_neighbors(rotterdam_hash)) {
        gs->config->transition_node_color(MICRO, node, 0xfffffffe);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("At every step, he’d ask:"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("is the current path to this node the shortest path so far?"), 3);
    gs->manager.transition(MICRO, "d", "0.02");
    gs->config->transition_edge_color(MICRO, rotterdam_hash, HashableString("The Hague").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_edge_color(MICRO, HashableString("The Hague").get_hash(), rotterdam_hash, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, this edge only has a weight of 1"), 3);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_edge_label(MICRO, rotterdam_hash, HashableString("The Hague").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("while the node's current cost is infinity."), 5);
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, rotterdam_hash, HashableString("The Hague").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("So this path is the shortest yet to that node,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("so Dijkstra updated its cost to 1."), 7);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    unordered_map<double, int> costs;
    unordered_set<double> visited;
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        costs[hash] = 1000;
    }
    costs[rotterdam_hash] = 0;
    attempt_relax_edge(gs, rotterdam_hash, HashableString("The Hague").get_hash(), costs);
    gs->render_microblock();

    stage_macroblock(FileBlock("Likewise, these nodes were updated to"), 3);
    gs->manager.transition(MICRO, "d", "0.03");
    gs->render_microblock();
    gs->config->splash_node(HashableString("Utrecht").get_hash());
    gs->config->splash_node(HashableString("Breda").get_hash());
    gs->config->splash_node(HashableString("'s-Hertogenbosch").get_hash());
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("3, 4 and 2."), 10);
    attempt_relax_edge(gs, rotterdam_hash, HashableString("Utrecht").get_hash(), costs);
    attempt_relax_edge(gs, rotterdam_hash, HashableString("Breda").get_hash(), costs);
    attempt_relax_edge(gs, rotterdam_hash, HashableString("'s-Hertogenbosch").get_hash(), costs);
    gs->render_microblock();

    stage_macroblock(FileBlock("After checking all of Rotterdam's neighboring nodes,"), 2);
    gs->manager.transition(MACRO, "d", "0.04");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra marked it as explored."), 1);
    put_in_visited_set(gs, rotterdam_hash, visited);

    stage_macroblock(FileBlock("Then, since this node had the lowest cost out of all the unexplored nodes,"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("Dijkstra explored it next."), 10);
    gs->manager.transition(MACRO, "d", "0.03");
    // Splash The Hague
    gs->config->splash_node(HashableString("The Hague").get_hash());
    set_camera_to_lat_long(gs, (netherlands_cities["Rotterdam"] + netherlands_cities["The Hague"]) / 2, false, MACRO);
    gs->render_microblock();
    gs->config->splash_node(HashableString("The Hague").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Like before, he checked each of its edges to reduce the neighbors’ costs."), 3);
    gs->render_microblock();
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

    stage_macroblock(FileBlock("Dijkstra could update this neighbor from infinity down to 6-"), 3);
    attempt_relax_edge(gs, hague_hash, HashableString("Amsterdam").get_hash(), costs);

    stage_macroblock(FileBlock("a total path length of 5+1."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But he couldn’t update this one."), 3);
    double utrecht_hash = HashableString("Utrecht").get_hash();
    attempt_relax_edge(gs, hague_hash, utrecht_hash, costs);

    stage_macroblock(FileBlock("The current path has a cost of 5,"), 7);
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "The Hague", "Utrecht"}, 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    splash_edge_path(MICRO, gs, {"Rotterdam", "The Hague", "Utrecht"});
    gs->render_microblock();

    stage_macroblock(FileBlock("but the node’s cost was 3."), 2);
    gs->config->fade_node_color(MICRO, utrecht_hash, 0xffffffff);
    gs->config->fade_node_color(MICRO, rotterdam_hash, visited_color);
    gs->config->fade_node_color(MICRO, hague_hash, 0xffffffff);
    gs->config->fade_edge_color(MICRO, rotterdam_hash, hague_hash, visited_color);
    gs->config->fade_edge_color(MICRO, hague_hash, utrecht_hash, visited_color);
    gs->render_microblock();
    gs->config->splash_node(utrecht_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("That meant there was already a shorter path:"), 3);
    gs->render_microblock();
    gs->config->splash_edge_label(MICRO, rotterdam_hash, utrecht_hash);
    trace_path(gs, {"Rotterdam", "Utrecht"}, 0xffff0000);
    visited.insert(hague_hash);
    restore_graph_colors(g, gs, visited, costs);

    stage_macroblock(FileBlock("this one directly from the source."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So there was no need to update the cost."), 2);
    put_in_visited_set(gs, hague_hash, visited);
    gs->manager.transition(MACRO, "d", "0.047");
    set_camera_to_lat_long(gs, center, false, MACRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("The next node to explore is this one with a cost of 2,"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("which would update its neighbour to 5."), 5);
    attempt_relax_all_edges(g, gs, HashableString("Breda").get_hash(), costs, visited);
    gs->render_microblock();
    stage_macroblock(FileBlock("Then this one with a cost of 3 would update its neighbours,"), 5);
    attempt_relax_all_edges(g, gs, HashableString("Utrecht").get_hash(), costs, visited);
    gs->render_microblock();
    stage_macroblock(FileBlock("and so on."), 5);
    attempt_relax_all_edges(g, gs, HashableString("'s-Hertogenbosch").get_hash(), costs, visited);
    gs->render_microblock();

    stage_macroblock(FileBlock("Sometimes he had to update a node's costs a few times."), 1);
    gs->manager.transition(MACRO, "d", "0.02");
    set_camera_to_lat_long(gs, netherlands_cities["'s-Hertogenbosch"], false, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, he first reached this node"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("through this path"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("with edge weights 4 and 3"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("for a total cost of 7."), 7);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("Eindhoven").get_hash());
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "'s-Hertogenbosch", "Eindhoven"}, 0xffff0000);
    splash_edge_path(MICRO, gs, {"Rotterdam", "'s-Hertogenbosch", "Eindhoven"});
    gs->render_microblock();
    restore_graph_colors(g, gs, visited, costs);

    stage_macroblock(FileBlock("But later, he found this path with cost 6."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("So he updated the node's cost again."), 12);
    splash_edge_path(MACRO, gs, {"Rotterdam", "Breda", "Tilburg", "Eindhoven"});
    gs->manager.transition(MACRO, "d", "0.04");
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "Breda", "Tilburg", "Eindhoven"}, 0xffff0000);
    restore_graph_colors(g, gs, visited, costs);
    gs->render_microblock();
    attempt_relax_all_edges(g, gs, HashableString("Tilburg").get_hash(), costs, visited);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(SilenceBlock(5), 25);
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
        if(closest == HashableString("Eindhoven").get_hash()){
            put_in_visited_set(gs, HashableString("Eindhoven").get_hash(), visited);
            continue;
        }
        attempt_relax_all_edges(g, gs, closest, costs, visited);
    }

    stage_macroblock(FileBlock("The first time Dijkstra reached Groningen,"), 6);
    attempt_relax_all_edges(g, gs, HashableString("Meppel").get_hash(), costs, visited);
    gs->config->transition_node_radius(MICRO, groningen_hash, 4);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("it had a cost of 19."), 3);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("But it didn’t have the lowest cost out of all the unexplored nodes."), 2);
    gs->config->transition_node_radius(MICRO, groningen_hash, 3);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("There was still this node with a cost of 16"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("which could be hiding a shorter path to our goal."), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->config->fade_node_color(MICRO, HashableString("Eindhoven").get_hash(), visited_color);
    gs->config->fade_node_color(MICRO, HashableString("Tilburg").get_hash(), visited_color);
    gs->config->fade_node_color(MICRO, HashableString("Breda").get_hash(), visited_color);
    gs->config->fade_node_color(MICRO, rotterdam_hash, visited_color);
    gs->config->fade_edge_color(MICRO, rotterdam_hash, HashableString("Breda").get_hash(), visited_color);
    gs->config->fade_edge_color(MICRO, HashableString("Breda").get_hash(), HashableString("Tilburg").get_hash(), visited_color);
    gs->config->fade_edge_color(MICRO, HashableString("Tilburg").get_hash(), HashableString("Eindhoven").get_hash(), visited_color);
    set_camera_to_lat_long(gs, center, false, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("So he continued exploring nodes in cost order,"), 8);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    attempt_relax_edge(gs, HashableString("Leeuwarden").get_hash(), groningen_hash, costs);
    put_in_visited_set(gs, HashableString("Leeuwarden").get_hash(), visited);
    gs->render_microblock();

    stage_macroblock(FileBlock("and found this path with cost 18."), 10);
    splash_edge_path(MACRO, gs, {"Rotterdam", "The Hague", "Amsterdam", "Almere", "Zwolle", "Meppel", "Leeuwarden", "Groningen"});
    gs->render_microblock();
    trace_path(gs, {"Rotterdam", "The Hague", "Amsterdam", "Almere", "Zwolle", "Meppel", "Leeuwarden", "Groningen"}, 0xffff8080);
    gs->render_microblock();
    restore_graph_colors(g, gs, visited, costs);

    stage_macroblock(FileBlock("Now, out of all the unexplored nodes,"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("Groningen had the next lowest cost."), 4);
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("So Dijkstra was confident he had found the shortest path."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Here's why."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("If the lowest cost among the unexplored nodes is 18,"), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();
    gs->config->transition_node_radius(MICRO, groningen_hash, 2);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(.2), 1);
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("then Dijkstra must have explored all the paths with cost"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("16, 14, 13,"), 22);
    // Length 16 path
    trace_path(gs, {"Rotterdam", "The Hague", "Amsterdam", "Almere", "Zwolle", "Meppel", "Leeuwarden"}, 0xffff6060);
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    gs->render_microblock();
    // Length 14 path
    trace_path(gs, {"Rotterdam", "The Hague", "Amsterdam", "Almere", "Zwolle", "Meppel"}, 0xffff6060);
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    gs->render_microblock();
    // Length 13 path
    trace_path(gs, {"Rotterdam", "The Hague", "Amsterdam", "Almere", "Zwolle", "Emmen"}, 0xffff6060);
    color_all_darkblue_except_endpoints(gs, g, rotterdam_hash, groningen_hash);
    gs->render_microblock();
    visited.insert(groningen_hash);

    stage_macroblock(FileBlock("Every path less than 18."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("And none of those paths reached Groningen,"), 3);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(groningen_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("It’s just like breadth-first search searching every step away from the source."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("By exploring from lowest to highest cost,"), true_city_costs.size() + 4);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    removable_cities = true_city_costs;
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

    stage_macroblock(FileBlock("Dijkstra’s algorithm guaranteed the shortest path to any target."), 1);
    // make nodes big again and relabel nodes
    gs->render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    gs->render_microblock();
}
