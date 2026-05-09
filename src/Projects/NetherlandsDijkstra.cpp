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
    {"Amsterdam", "The Hague"},
    {"Amsterdam", "Utrecht"},
};

vector<pair<string, string>> netherlands_edges_2 = {
    {"Amsterdam", "Zwolle"},
    {"The Hague", "Rotterdam"},
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
    gs->manager.transition(MACRO, "globe_opacity", "1");
    set_camera_to_lat_long(gs, vec2(52.5, 5.5), true, MACRO);
    gs->manager.set({
        {"physics_multiplier","0"},
        {"d", ".07"},
    });

    stage_macroblock(FileBlock("In other words, a shortest path algorithm."), 1);
    gs->render_microblock();

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

    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        gs->transition_node_position(MICRO, HashableString(city).get_hash(), position);
    }
    stage_macroblock(FileBlock("So let's add distances."), 1);
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

    stage_macroblock(FileBlock("But until he actually explored the graph, he didn’t know the shortest distance to any node yet."), 6);

    stage_macroblock(FileBlock("So at the beginning, the source's cost was zero,"), 2);
    int step = 0;
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("and every other node was infinity."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("There’s two things to keep in mind:"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("One, Dijkstra’s algorithm will always explore nodes from lowest to highest cost."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("And two, it updates costs as it finds shorter and shorter paths."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("That’s why the other nodes start at infinity. It hasn’t explored any paths yet. Let’s see it in action."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Rotterdam is first, so the algorithm checks all of its edges."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("The current path from A to B costs 1, which is less than B’s current cost of infinity. It updates, or relaxes, B’s cost to 1."), 3);
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("A also relaxes C to .7, D to .3, and E to 1.3."), 12);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("That’s all of A’s neighbors, so the algorithm marks A as explored."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("B is next since it has the lowest cost out of all the unexplored nodes."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("B relaxes G’s cost to 6 and E’s cost to 3."), 6);
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("Remember, the cost is the distance from the source, so for G and E, we have to also add the cost to get to B."), 2);
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    return;

    stage_macroblock(FileBlock("But B can’t relax C since the path A B C costs 6, when C’s current cost is 3. The algorithm only keeps the shortest paths."), 1);
    stage_macroblock(FileBlock("This continues for the rest of the graph. If there’s any ties in the lowest cost, the algorithm can explore them in any order."), 1);
    stage_macroblock(FileBlock("And if the algorithm runs into a node it’s relaxed once, like going from E to G, it still compares the current path to the node’s current cost. In this case, the path A - B - E - G is shorter than A - B - G. So the algorithm updates G’s cost."), 1);
    stage_macroblock(FileBlock("When the target M has the next lowest cost, or it’s next up to be explored, the algorithm has built the shortest path up to M. It returns the shortest path length — 10. This is also M’s final cost."), 1);
    stage_macroblock(FileBlock("And if we mark what nodes are able to relax others, a sort of predecessor list, we can easily build directions for the shortest paths."), 1);
    stage_macroblock(FileBlock("Here’s a few more examples of Dijkstra’s running through different graphs. Just like breadth-first, there’s a search-frontier that slowly spreads through the nodes. But Dijkstra’s frontier jumps around in different directions based on the edge weights."), 1);
}
