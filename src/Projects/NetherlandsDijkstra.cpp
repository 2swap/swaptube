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
};

vector<tuple<string, string, int>> netherlands_edges = {
    {"Rotterdam", "Utrecht", 3},
    {"Rotterdam", "Breda", 2},
    {"Breda", "Tillburg", 3},
    {"Tillburg", "Eindhoven", 1},
    {"Eindhoven", "'s-Hertogenbosch", 2},
    {"'s-Hertogenbosch", "Utrecht", 2},
    {"Utrecht", "Arnhem", 4},
    {"Arnhem", "Zwolle", 5},
    {"Zwolle", "Emmen", 2},
    {"Zwolle", "Meppel", 3},
    {"Meppel", "Groningen", 5},
    {"Meppel", "Leeuwarden", 2},
    {"Amsterdam", "The Hague", 5},
    {"Amsterdam", "Utrecht", 3},
    {"Amsterdam", "Zwolle", 5},
    {"The Hague", "Rotterdam", 1},
    {"Tillburg", "'s-Hertogenbosch", 1},
    {"Groningen", "Leeuwarden", 2},
    {"The Hague", "Utrecht", 4},
    {"Rotterdam", "'s-Hertogenbosch", 4},
    {"'s-Hertogenbosch", "Arnhem", 2},
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

void run_dijkstra(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double start, double goal, int up_to_step) {
    std::unordered_set<double> visited;
    std::unordered_map<double, double> costs;
    std::unordered_set<double> open_set;
    std::unordered_map<double, double> came_from;

    open_set.insert(start);
    gs->config->transition_node_color(MICRO, start, 0xffff0000);
    if(--up_to_step == 0) return;

    costs[start] = 0;
    if(g->size() < 10000) gs->config->transition_node_label(MICRO, start, "0");
    if(--up_to_step == 0) return;

    for(auto& [hash, node] : g->nodes) {
        if(hash == start) continue;
        costs[hash] = std::numeric_limits<double>::infinity();
        if(g->size() < 10000) gs->config->transition_node_label(MICRO, hash, "\\infty");
    }
    if(--up_to_step == 0) return;

    while(open_set.size() > 0) {
        // Find node in open set with lowest cost
        double current = -1;
        double current_cost = std::numeric_limits<double>::infinity();
        for(double hash : open_set) {
            if(costs[hash] < current_cost) {
                current_cost = costs[hash];
                current = hash;
            }
        }

        if(current == goal) {
            cout << "Reached goal!" << endl;
            break;
        }

        open_set.erase(current);
        gs->config->transition_node_color(MICRO, current, 0xff0000ff);
        if(--up_to_step == 0) return;

        unordered_set<double> neighbors = g->get_neighbors(current);

        for(double neighbor : neighbors) {
            if(visited.find(neighbor) != visited.end()) {
                continue;
            }
            double weight = length(g->nodes.find(current)->second.position - g->nodes.find(neighbor)->second.position);
            double tentative_cost = costs[current] + weight;
            gs->config->transition_edge_color(MICRO, current, neighbor, 0xffff0000);
            if(--up_to_step == 0) return;

            if(tentative_cost < costs[neighbor]) {
                came_from[neighbor] = current;
                costs[neighbor] = tentative_cost;
                if(g->size() < 10000) gs->config->transition_node_label(MICRO, neighbor, to_string_with_precision(tentative_cost, 1));
                if(open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                    gs->config->transition_node_color(MICRO, neighbor, 0xffff0000);
                }
            }
            if(--up_to_step == 0) return;

            gs->config->fade_edge_color(MICRO, neighbor, current, 0xffffffff);
            if(--up_to_step == 0) return;
        }
        visited.insert(current);
        gs->config->transition_node_color(MICRO, current, 0xff00ff00);
        if(--up_to_step == 0) return;
    }
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

    stage_macroblock(FileBlock("Dijkstra realized that it wasn’t the number of edges to the source that mattered, it was the distance to the source."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So instead, he could explore the closest node to the source, and then the next closest node, and so on."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("He’d track the shortest known distance to the node, called its cost. That way, he could always explore nodes from lowest to highest cost, or closest to furthest from the source."), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->manager.transition(MICRO, "points_radius_multiplier", "2");
    gs->render_microblock();
    gs->manager.transition(MICRO, "points_radius_multiplier", "1");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    return;

    stage_macroblock(FileBlock("But until he actually explored the graph, he didn’t know the shortest distance to any node yet."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So at the beginning, the source's cost was zero,"), 2);
    int step = 0;
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("and every other node was infinity."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("Next, when Dijkstra explored each neighboring node,"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("he’d ask: is the current path the shortest one yet?"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("For example, this edge only has a weight of 1 while the node’s current cost is infinity."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("This path is the shortest yet to that node [1 < inf], so Dijkstra updated its cost to 1."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("After checking all of Rotterdam’s edges and updating the neighbors’ costs,"), 1);
    stage_macroblock(FileBlock("Dijkstra marked it as explored. If this was breadth first search,"), 1);
    stage_macroblock(FileBlock("he could pick any neighbor to explore next. But he wanted to prioritize closer nodes."), 1);
    stage_macroblock(FileBlock("So he explored the nodes from lowest to highest cost."), 1);
    stage_macroblock(FileBlock("With a cost of 1, this node was next."), 1);
    stage_macroblock(FileBlock("Like before, he checked each of its edges to reduce the neighbors’ costs."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("While Dijkstra could update one neighbor, he couldn’t update this one."), 1);
    stage_macroblock(FileBlock("This path has a cost of 5, but the node’s cost was 3."), 1);
    stage_macroblock(FileBlock("There was already a shorter path, so no need to update it."), 1);
    stage_macroblock(FileBlock("This continued for the rest of the graph."), 1);
    stage_macroblock(FileBlock("Sometimes he had to update a node's costs a few times."), 1);
    stage_macroblock(FileBlock("For example, he first reached this node through this path with a cost of 6."), 1);
    stage_macroblock(FileBlock("But later, he found a shorter path of cost 5 and updated the node’s cost again."), 1);
    stage_macroblock(FileBlock("The first time Dijkstra reached Groningen, it had a cost of 18."), 1);
    stage_macroblock(FileBlock("But it didn’t have the lowest cost out of all the unexplored nodes."), 1);
    stage_macroblock(FileBlock("There could still be a shorter path, just like earlier."), 1);
    stage_macroblock(FileBlock("He continued exploring nodes in cost order, and found this path with cost 17."), 1);
    stage_macroblock(FileBlock("Now out of all the unexplored nodes, Groningen had the next lowest cost."), 1);
    stage_macroblock(FileBlock("Dijkstra was confident he had found the shortest path."), 1);
    stage_macroblock(FileBlock("That’s because on a road network, the cost always increases along a path."), 1);
    stage_macroblock(FileBlock("So if the lowest cost among the unexplored nodes was 17,"), 1);
    stage_macroblock(FileBlock("Dijkstra must have explored all the paths with cost 16, 15, 16 and so on."), 1);
    stage_macroblock(FileBlock("And none of those paths reached Groningen, otherwise he would’ve updated its cost earlier."), 1);
    stage_macroblock(FileBlock("It’s just like the levels of breadth-first search."), 1);
    stage_macroblock(FileBlock("By exploring from lowest to highest cost,"), 1);
    stage_macroblock(FileBlock("Dijkstra’s algorithm guaranteed the shortest path to any target."), 1);
    stage_macroblock(FileBlock("To get directions for the shortest path,"), 1);
    stage_macroblock(FileBlock("we just need to keep track of what nodes are able to relax others — a sort of predecessor list."), 1);
    stage_macroblock(FileBlock("Then at the end, we can easily work backwards to get the nodes in order."), 1);

}
