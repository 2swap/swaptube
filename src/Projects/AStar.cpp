#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"
uint32_t opaque_white = 0x20ffffff;

// lat long map

// File format:
// File starts with line NODES
// Then nodes are listed: id (integer), latitude (float), longitude (float)
// Then line EDGES
// Then edges are listed: node1 (integer), node2 (integer)
// Ignore any nodes or edges that are outside the given radius from the center point
void load_graph_from_file(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, vec2 center, float radius) {
    ifstream file("io_in/graph_speeds.txt");
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

// Run dijkstra's algorithm up until some node within max_dist of the goal is added to the visited set.
// Color all searched edges blue.
void run_large_dijkstra(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double start, double goal, double max_dist, float heuristic_mult) {
    gs->config->set_all_edge_colors(0x10ffffff);
    std::unordered_set<double> visited;
    std::unordered_map<double, double> costs;

    std::unordered_set<double> open_set;

    std::unordered_map<double, double> came_from;

    open_set.insert(start);
    costs[start] = 0;
    vec4 start_pos = g->nodes.find(start)->second.position;

    for(auto& [hash, node] : g->nodes) {
        if(hash == start) continue;
        costs[hash] = std::numeric_limits<double>::infinity();
    }

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

        vec4 current_pos = g->nodes.find(current)->second.position;

        if (length(current_pos - start_pos) > max_dist) {
            return;
        }
        if (current == goal) {
            cout << "Reached goal!" << endl;
            gs->config->transition_node_color(MICRO, current, 0xff00ff01);
            // Color the path from current to start green
            double path_node = current;
            while(path_node != start) {
                double parent = came_from[path_node];
                gs->config->set_edge_color(path_node, parent, 0xff00ffff);
                path_node = parent;
            }
            return;
        }

        open_set.erase(current);

        unordered_set<double> neighbors = g->get_neighbors(current);

        for(double neighbor : neighbors) {
            if(visited.find(neighbor) != visited.end()) {
                continue;
            }
            double weight = length(g->nodes.find(current)->second.position - g->nodes.find(neighbor)->second.position);
            weight += length(g->nodes.find(neighbor)->second.position - g->nodes.find(goal)->second.position) * heuristic_mult;
            double tentative_cost = costs[current] + weight;

            if(tentative_cost < max_dist)
                gs->config->set_edge_color(current, neighbor, 0x10ff8080);

            if(tentative_cost < costs[neighbor]) {
                costs[neighbor] = tentative_cost;
                came_from[neighbor] = current;

                if(open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                }
            }
        }
        visited.insert(current);
    }
}

void heuristic_slide(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double zoo_hash, double factor, TransitionType tt) {
    for(auto& [hash, node] : g->nodes) {
        double distance_to_zoo = length(node.position - g->nodes.find(zoo_hash)->second.position);
        vec4 new_position = normalize(node.position) * (1 + distance_to_zoo * factor / 5);
        gs->transition_node_position(tt, hash, new_position);
    }
}

void render_video() {
    vec2 newark_lat_long = vec2(40.694669192970665, -74.18676933576879);
    vec2 zoo_lat_long = vec2(40.767665443249214, -73.97196914550813);

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"globe_opacity", "0"},
        {"d", ".005"},
    });

    // Fade globe to opacity 1
    vec2 center = newark_lat_long;
    set_camera_to_lat_long(gs, center, true, MICRO);
    stage_macroblock(SilenceBlock(1), 1);
    double newark_hash;
    double zoo_hash;
    if(rendering_on()) {
        load_graph_from_file(g, gs, newark_lat_long, 0.21);
        newark_hash = get_nearest_node_in_graph(g, newark_lat_long);
        zoo_hash = get_nearest_node_in_graph(g, zoo_lat_long);
        gs->config->set_node_radius(newark_hash, 1);
        gs->config->set_node_radius(zoo_hash, 1);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("We'd like to prioritize nodes that are closer to the Zoo."), 1);
    for(auto& [hash, node] : g->nodes) {
        unordered_set<double> neighbors = g->get_neighbors(hash);
        for(double neighbor : neighbors) {
            double distance_to_zoo = length(node.position - g->nodes.find(zoo_hash)->second.position);
            double distance_zoo_to_newark = length(g->nodes.find(zoo_hash)->second.position - g->nodes.find(newark_hash)->second.position);
            double ratio = .25 * distance_to_zoo / distance_zoo_to_newark;
            uint32_t color = rainbow(ratio);
            gs->config->fade_edge_color(MICRO, hash, neighbor, color);
        }
    }
    gs->render_microblock();
    gs->config->set_all_edge_colors(opaque_white);
    gs->config->set_all_node_colors(0x00000000);

    /*
    stage_macroblock(FileBlock("Using longitudes and latitudes, we can easily calculate the straight line distance between any node and the target."), 1);
    stage_macroblock(FileBlock("We'll order nodes by their cost plus this straight line distance."), 1);
    stage_macroblock(FileBlock("Nodes in the opposite direction won't be explored early on."), 1);

    // Re-do Dijkstra's
    */
    stage_macroblock(FileBlock("Let's see these two algorithms side by side."), 1);
    gs->config->fade_all_edge_colors(MICRO, 0x10ffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra’s search frontier spreads out in all directions."), 100);
    int max_dist = 0;
    int increment = 0.00003;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("But this modified Dijkstra's"), 1);
    gs->config->fade_all_edge_colors(MICRO, 0x10ffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("also called A* immediately heads towards Manhattan."), 150);
    max_dist = 0;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 1);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("It only checks around 7,000 nodes — that’s almost a 10x improvement!"), 1);
    gs->render_microblock();

    // Transition all nodes' positions to scale as a function of their distance to the zoo.
    stage_macroblock(FileBlock("Each node’s height is its straight line distance to the target. This distance is also called a heuristic."), 1);
    gs->manager.transition(MICRO, "theta", ".8");
    gs->manager.transition(MICRO, "d", ".01");
    heuristic_slide(g, gs, zoo_hash, 1, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("Now we can really see how the heuristic funnels the search directly towards Central Park."), 1);
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But say there’s a direct route to Central Park that starts just west of Newark. Now the shortest path to the zoo “illogically” goes west first."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("When the heuristic is zero, A* is the same as Dijkstra’s,"), 150);
    heuristic_slide(g, gs, zoo_hash, 0, MICRO);
    gs->render_microblock();
    max_dist = 1000;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0);
        gs->render_microblock();
    }
    stage_macroblock(FileBlock("so it still finds the shortest path."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("As we raise the heuristic, our search becomes more and more directed."), 150);
    heuristic_slide(g, gs, zoo_hash, 2, MACRO);
    float total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 2 * (1 - remaining_microblocks_in_macroblock / total_microblocks));
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("At some point, it will head towards Manhattan so aggressively, it doesn't explore west at all and returns the wrong path."), 150);
    heuristic_slide(g, gs, zoo_hash, 5, MACRO);
    total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 2 + 3 * (1 - remaining_microblocks_in_macroblock / total_microblocks));
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("If we want A* to always find the shortest path, the heuristic needs to fulfill two conditions. First, it must underestimate the overall distance between a node and target."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("And second, the difference between two heuristics must underestimate the true cost between those nodes. These conditions make sure the heuristic narrows the search without missing any sneaky paths."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("To optimize for time, we divide the straight line distance by the maximum speed allowed on the graph. This gives us the minimum travel time."), 150);
    heuristic_slide(g, gs, zoo_hash, .1, MACRO);
    total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, .1 + 4.9 * (remaining_microblocks_in_macroblock / total_microblocks)) ;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("We have to use the maximum speed and not the speed limit since there could be multiple speed limits from one node to the target. The heuristic needs to be an underestimate."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But in this case, it’s an extreme underestimate. It flattens out the graph so much, the search ends up looking more like Dijkstra’s."), 1);
    gs->render_microblock();
}
