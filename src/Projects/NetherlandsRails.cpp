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

void bfs_simul(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double>& border, unordered_set<double>& next_border, unordered_set<double>& visited, int depth) {
    double node = *border.begin();
    // Find node with least neighbors in the border
    for(double candidate : border) {
        if(g->get_neighbors(candidate).size() < g->get_neighbors(node).size()) {
            node = candidate;
        }
    }
    gs->render_microblock();
    unordered_set<double> neighbors = g->get_neighbors(node);
    for(double neighbor : neighbors) {
        if(visited.find(neighbor) != visited.end()) continue;
        gs->config->transition_edge_color(MICRO, node, neighbor, colors_by_depth[depth]);
    }
    gs->render_microblock();
    for(double neighbor : neighbors) {
        if(visited.find(neighbor) != visited.end()) continue;
        if (next_border.find(neighbor) == next_border.end() && border.find(neighbor) == border.end()) {
            next_border.insert(neighbor);
            gs->config->transition_node_color(MICRO, neighbor, colors_by_depth[depth]);
        }
    }
    gs->render_microblock();
    border.erase(node);
    visited.insert(node);
    gs->config->transition_node_color(MICRO, node, 0xffffffff);
    gs->render_microblock();

    if(border.size() == 0) {
        border = next_border;
        next_border.clear();
    }
}

void bfs(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double>& border, unordered_set<double>& next_border, unordered_set<double>& visited, int depth, bool simul) {
    if(simul) {
        bfs_simul(g, gs, border, next_border, visited, depth);
        return;
    }
    double node = *border.begin();
    gs->manager.transition(MICRO, {
        {"x", to_string(g->nodes.find(node)->second.position.x)},
        {"y", to_string(g->nodes.find(node)->second.position.y)},
        {"z", to_string(g->nodes.find(node)->second.position.z)},
    });
    unordered_set<double> neighbors = g->get_neighbors(node);
    for(double neighbor : neighbors) {
        if(visited.find(neighbor) != visited.end()) continue;
        gs->config->transition_edge_color(MICRO, node, neighbor, colors_by_depth[depth]);
        gs->render_microblock();
        if (next_border.find(neighbor) == next_border.end() && border.find(neighbor) == border.end()) {
            next_border.insert(neighbor);
            gs->config->transition_node_color(MICRO, neighbor, colors_by_depth[depth]);
            gs->render_microblock();
        }
    }
    border.erase(node);
    visited.insert(node);
    gs->config->transition_node_color(MICRO, node, 0xff00ff00);
    gs->render_microblock();

    if(border.size() == 0) {
        border = next_border;
        next_border.clear();
    }
}

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
    gs->config->transition_node_color(MICRO, groningen_hash, 0xffff0001);
    gs->render_microblock();

    stage_macroblock(FileBlock("this path seems obvious."), path.size());
    set_camera_to_lat_long(gs, netherlands_cities["Groningen"], false, MACRO);
    for(int i = 0; i < path.size() - 1; i++) {
        string city = path[i];
        string neighbor = path[i+1];
        gs->config->transition_edge_color(MICRO, HashableString(city).get_hash(), HashableString(neighbor).get_hash(), 0xffff0000);
        gs->render_microblock();
        gs->config->transition_node_color(MICRO, HashableString(neighbor).get_hash(), 0xffff0000);
    }
    gs->render_microblock();

    // Add more nodes
    stage_macroblock(FileBlock("And it is the shortest path, but what about now?"), netherlands_edges_2.size() + 6);
    set_camera_to_lat_long(gs, vec2(52.5, 5.5), false, MACRO);
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

    // Slide 9

    stage_macroblock(FileBlock("Here's a possible method:"), 1);
    uint32_t edge_dark = 0xff808080;
    gs->config->fade_all_node_colors(MICRO, edge_dark);
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->render_microblock();

    bool simul = true;
    stage_macroblock(FileBlock("Starting from the source, we explore all of its neighboring nodes to see if the target is there."), simul?5:10);
    gs->manager.transition(MICRO, {
        {"x", to_string(g->nodes.find(HashableString("Rotterdam").get_hash())->second.position.x)},
        {"y", to_string(g->nodes.find(HashableString("Rotterdam").get_hash())->second.position.y)},
        {"z", to_string(g->nodes.find(HashableString("Rotterdam").get_hash())->second.position.z)},
    });
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);
    gs->render_microblock();

    unordered_set<double> border;
    unordered_set<double> next_border;
    unordered_set<double> visited;
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    bfs(g, gs, border, next_border, visited, 0, simul);

    stage_macroblock(FileBlock("If it isn't, we move on to those explored nodes and then check their neighbors for the target."), simul?16:17);
    set_camera_to_lat_long(gs, netherlands_cities["Utrecht"], false, MACRO);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);

    stage_macroblock(FileBlock("If we don't find it, we expand again."), simul?12:7);
    set_camera_to_lat_long(gs, netherlands_cities["Arnhem"], false, MACRO);
    bfs(g, gs, border, next_border, visited, 2, simul);
    bfs(g, gs, border, next_border, visited, 2, simul);
    bfs(g, gs, border, next_border, visited, 2, simul);

    stage_macroblock(FileBlock("And we keep expanding out until we reach the target node."), simul?16:12);
    set_camera_to_lat_long(gs, netherlands_cities["Meppel"], false, MACRO);
    bfs(g, gs, border, next_border, visited, 3, simul);
    bfs(g, gs, border, next_border, visited, 3, simul);

    bfs(g, gs, border, next_border, visited, 4, simul);
    bfs(g, gs, border, next_border, visited, 4, simul);

    stage_macroblock(FileBlock("This algorithm is known as breadth first search."), 1);
    gs->render_microblock();
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->config->fade_all_node_colors(MICRO, edge_dark);

    stage_macroblock(FileBlock("It will always find the shortest path because it checks all of the nodes at every level."), simul?4:9);
    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    bfs(g, gs, border, next_border, visited, 0, simul);

    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    stage_macroblock(FileBlock("If there was a path from Rotterdam to Groningen in two steps,"), simul?4:9);
    bfs(g, gs, border, next_border, visited, 0, simul);
    stage_macroblock(FileBlock("we would’ve found it on the second iteration, so six steps must be the shortest path."), simul?22:17);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);

    gs->manager.set("physics_multiplier", "1");
    gs->manager.set("repel", ".01");
    gs->manager.set("decay", "0.2");
    gs->manager.transition(MICRO, "decay", "0.5");
    return;

    stage_macroblock(FileBlock("But there’s a problem. This algorithm values all of the edges the same. Just one step."), 1);
    gs->render_microblock();
    gs->manager.set("physics_multiplier", "0");

    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        gs->transition_node_position(MICRO, HashableString(city).get_hash(), position);
    }
    stage_macroblock(FileBlock("So let's adjust that map, and add weights to represent distance."), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 2);
    gs->manager.transition(MICRO, "edge_weights_size", "1");
    gs->render_microblock();
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, "");
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("Now the shortest path is much harder to figure out."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra realized that it wasn’t the number of edges between a node and the source that mattered, it was their distance to the source."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So instead of exploring nodes by their level, he could explore all the nodes close to the source, and then the next closest nodes, and so on — ordered by their distance."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra’s algorithm works like this. Each node has a cost, or how far it is from the source."), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->manager.transition(MICRO, "points_radius_multiplier", "2");
    gs->render_microblock();
    gs->manager.transition(MICRO, "points_radius_multiplier", "1");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("At the start, the source has a cost of zero,"), 2);
    int step = 0;
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("and every other node has a cost of infinity."), 1);
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

// File format:
// File starts with line NODES
// Then nodes are listed: id (integer), latitude (float), longitude (float)
// Then line EDGES
// Then edges are listed: node1 (integer), node2 (integer)
// Ignore any nodes or edges that are outside the given radius from the center point
void load_graph_from_file(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, vec2 center, float radius) {
    ifstream file("io_in/graph.txt");
    string line;
    enum Section { NONE, NODES, EDGES };
    Section section = NONE;
    int node_count = 0;
    int edge_count = 0;
    while(getline(file, line)) {
        if(line == "NODES") {
            section = NODES;
            continue;
        } else if(line == "EDGES") {
            section = EDGES;
            continue;
        }
        if(section == NODES) {
            stringstream ss(line);
            int id;
            float lat, longi;
            ss >> id >> lat >> longi;
            double hash = HashableString(to_string(id)).get_hash();
            vec4 position = lat_long_to_xyz(vec2(lat, longi));
            if (length(vec2(lat, longi) - center) > radius) continue;
            g->add_node(new HashableString(to_string(id)));
            g->move_node(hash, position);
            gs->config->set_node_radius(hash, 0);
            node_count++;
        } else if(section == EDGES) {
            stringstream ss(line);
            int id1, id2;
            double weight;
            ss >> id1 >> id2;
            double hash1 = HashableString(to_string(id1)).get_hash();
            double hash2 = HashableString(to_string(id2)).get_hash();
            g->add_edge(hash1, hash2);
            edge_count++;
        }
    }
    cout << "Loaded graph with " << node_count << " nodes and " << edge_count << " edges." << endl;
}

double get_nearest_node_in_graph(shared_ptr<Graph> g, vec2 lat_long) {
    double nearest_node = -1;
    double nearest_distance = std::numeric_limits<double>::infinity();
    vec3 node_xyz = lat_long_to_xyz(lat_long);
    for(auto& [hash, node] : g->nodes) {
        vec3 position = node.position;
        double distance = length(position - node_xyz);
        if(distance < nearest_distance) {
            nearest_distance = distance;
            nearest_node = hash;
        }
    }
    return nearest_node;
}
