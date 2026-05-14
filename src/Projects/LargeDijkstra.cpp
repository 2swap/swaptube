#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

uint32_t opaque_white = 0x20ffffff;
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
            gs->config->set_node_color(hash, 0x00000000);
            node_count++;
        } else if(section == EDGES) {
            stringstream ss(line);
            int id1, id2;
            double weight;
            ss >> id1 >> id2;
            double hash1 = HashableString(to_string(id1)).get_hash();
            double hash2 = HashableString(to_string(id2)).get_hash();
            if(g->nodes.find(hash1) == g->nodes.end() || g->nodes.find(hash2) == g->nodes.end()) {
                continue;
            }
            g->add_edge(hash1, hash2);
            edge_count++;
        }
    }
    cout << "Loaded graph with " << node_count << " nodes and " << edge_count << " edges." << endl;
}

// Defined as set of nodes whose squared distance to zoo is more than twice the distance from the zoo to Newark
void get_new_jersey_nodes(shared_ptr<Graph> g, unordered_set<double>& new_jersey_nodes) {
    vec3 newark = lat_long_to_xyz(vec2(40.694669192970665, -74.18676933576879));
    vec3 zoo = lat_long_to_xyz(vec2(40.767665443249214, -73.97196914550813));
    double separation = length(newark - zoo);
    for(auto& [hash, node] : g->nodes) {
        double dist_to_zoo = length(node.position - zoo);
        if(dist_to_zoo > separation * 1.2) {
            new_jersey_nodes.insert(hash);
        }
    }
}

void get_staten_island_nodes(shared_ptr<Graph> g, unordered_set<double>& staten_island_nodes) {
    vec2 bayonne_lat_long = vec2(40.632, -74.145);
    vec2 goethals_lat_long = vec2(40.629, -74.185);
    vec2 verrazano_lat_long = vec2(40.602, -74.060);
    vec2 outer_bay_lat_long = vec2(40.525, -74.237);
    vector<vec2> staten_island_bridges = {bayonne_lat_long, goethals_lat_long, verrazano_lat_long, outer_bay_lat_long};
    vector<vec3> staten_island_bridges_xyz;
    for(vec2 bridge : staten_island_bridges) {
        staten_island_bridges_xyz.push_back(lat_long_to_xyz(bridge));
        cout << "Staten Island bridge at lat long: " << bridge.x << ", " << bridge.y << " has xyz: " << staten_island_bridges_xyz.back().x << ", " << staten_island_bridges_xyz.back().y << ", " << staten_island_bridges_xyz.back().z << endl;
    }
    vec2 staten_island_center = vec2(40.581, -74.147);
    double closest_node_to_center = get_nearest_node_in_graph(g, staten_island_center);
    // Flood fill from center of staten island outwards until we hit a bridge, adding nodes to staten_island_nodes
    unordered_set<double> visited;
    queue<double> q;
    q.push(closest_node_to_center);
    while(!q.empty()) {
        double current = q.front();
        q.pop();
        if(visited.find(current) != visited.end()) continue;
        visited.insert(current);
        vec3 current_pos = g->nodes.find(current)->second.position;
        bool is_bridge = false;
        for(const vec3& bridge : staten_island_bridges_xyz) {
            if(length(g->nodes.find(current)->second.position - bridge) < 0.00008) {
                is_bridge = true;
                break;
            }
        }
        if(is_bridge) continue;
        staten_island_nodes.insert(current);
        unordered_set<double> neighbors = g->get_neighbors(current);
        for(double neighbor : neighbors) {
            if(visited.find(neighbor) == visited.end()) {
                q.push(neighbor);
            }
        }
    }
}

// Run dijkstra's algorithm up until some node within max_dist of the goal is added to the visited set.
// Color all searched edges blue.
bool run_large_dijkstra(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double start, double goal, double max_dist, float heuristic_mult, unordered_set<double> highlighted_nodes = {}) {
    cout << "Running large dijkstra with max_dist " << max_dist << " and heuristic_mult " << heuristic_mult << endl;
    //gs->config->set_all_edge_colors(opaque_white);
    std::unordered_set<double> visited;
    std::unordered_map<double, double> costs;

    std::unordered_set<double> open_set;

    std::unordered_map<double, double> came_from;

    bool transition_not_fade = highlighted_nodes.size() == 0;
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
            return false;
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
            return true;
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

            if(tentative_cost < max_dist) {
                int color = 0x10ff8080;
                if(highlighted_nodes.find(neighbor) != highlighted_nodes.end()) {
                    color = 0x2000ff00;
                }
                if(transition_not_fade) {
                    gs->config->transition_edge_color(MICRO, current, neighbor, color);
                } else {
                    gs->config->fade_edge_color(MICRO, current, neighbor, color);
                }
            }

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
    return false;
}

void render_video() {
    vec2 newark_lat_long = vec2(40.694669192970665, -74.18676933576879);
    vec2 zoo_lat_long = vec2(40.767665443249214, -73.97196914550813);

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"globe_opacity", "1"},
        {"d", ".07"},
        {"points_opacity", "0"},
        {"lines_opacity", "0"},
    });

    // Fade globe to opacity 1
    vec2 center = newark_lat_long;
    set_camera_to_lat_long(gs, center, true, MICRO);
    stage_macroblock(SilenceBlock(1), 1);
    double newark_hash;
    double zoo_hash;
    if(rendering_on()) {
        load_graph_from_file(g, gs, newark_lat_long, 100);
        newark_hash = get_nearest_node_in_graph(g, newark_lat_long);
        zoo_hash = get_nearest_node_in_graph(g, zoo_lat_long);
        gs->config->set_node_radius(newark_hash, 1);
        gs->config->set_node_radius(zoo_hash, 1);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("Let's say I want to get from Newark Airport in New Jersey over to the Central Park Zoo."), 3);
    gs->config->set_all_edge_colors(opaque_white);
    gs->config->set_all_node_colors(0x00000000);
    gs->manager.transition(MICRO, "globe_opacity", "0.2");
    gs->manager.transition(MICRO, "points_opacity", "1");
    gs->manager.transition(MICRO, "lines_opacity", "1");
    gs->manager.transition(MICRO, "d", ".01");
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, newark_hash, 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, zoo_hash, 0xff00ff00);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->manager.transition(MACRO, "d", ".002");
    gs->render_microblock();

    if(rendering_on() && (newark_hash == -1 || zoo_hash == -1)) {
        cout << "Newark hash: " << newark_hash << " Zoo hash: " << zoo_hash << endl;
        throw runtime_error("Could not find nearest node for Newark or Zoo.");
        return;
    }
    if(rendering_on() && newark_hash == zoo_hash) {
        throw runtime_error("Nearest node for Newark and Zoo is the same. Check if the graph is loaded correctly and if the nearest node function is working.");
        return;
    }

    int chunk = 50;
    gs->manager.transition(MACRO, "d", ".004");
    stage_macroblock(FileBlock("Dijkstra’s algorithm checks all the ten minute journeys,"), chunk * 2);
    double max_dist = 0;
    double increment = 0.00002 * 50. / chunk;
    for(int i = 0; i < chunk; i++) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0);
        max_dist += increment;
        gs->render_microblock();
    }
    for(int i = 0; i < chunk; i++) {
        gs->render_microblock();
    }

    gs->manager.transition(MACRO, "d", ".01");
    stage_macroblock(FileBlock("and then all the twenty minute journeys,"), chunk * 2);
    for(int i = 0; i < chunk; i++) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0);
        max_dist += increment;
        gs->render_microblock();
    }
    for(int i = 0; i < chunk; i++) {
        gs->render_microblock();
    }

    bool goal = false;
    stage_macroblock(FileBlock("and so on until it reaches all the forty minute journeys, including the Zoo."), chunk * 2);
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !goal) goal = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("The search frontier covers over 65,000 nodes and includes places that are way off,"), 1);
    gs->render_microblock();

    cout << "Finding nodes in Staten Island..." << endl;
    unordered_set<double> staten_island_nodes;
    if(rendering_on()) get_staten_island_nodes(g, staten_island_nodes);
    cout << "Finding nodes in New Jersey..." << endl;
    unordered_set<double> new_jersey_nodes;
    if(rendering_on()) get_new_jersey_nodes(g, new_jersey_nodes);
    cout << "Staten Island nodes: " << staten_island_nodes.size() << " New Jersey nodes: " << new_jersey_nodes.size() << endl;

    stage_macroblock(FileBlock("like Staten Island and large swaths of New Jersey before it even hits Central Park."), 9);
    gs->manager.transition(MICRO, "d", ".004");
    set_camera_to_lat_long(gs, vec2(40.584430, -74.143991), false, MICRO);
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 1000, 0, staten_island_nodes);
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 1000, 0, {-1.234567});
    gs->render_microblock();
    set_camera_to_lat_long(gs, vec2(40.657, -74.241), false, MICRO);
    gs->manager.transition(MICRO, "d", ".006");
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 1000, 0, new_jersey_nodes);
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 1000, 0, {-1.234567});
    gs->render_microblock();
    gs->manager.transition(MICRO, "d", ".01");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("But even though it searched in illogical directions, the runtime was around 91 milliseconds. Which is incredibly fast."), 1);
    gs->render_microblock();
}
