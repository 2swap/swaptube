#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

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
                staten_island_nodes.insert(current);
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
    // The bridges themselves should also be considered part of Staten Island, so add any nodes within 0.00008 of the bridges to staten_island_nodes
    for(const vec3& bridge : staten_island_bridges_xyz) {
        for(auto& [hash, node] : g->nodes) {
            if(length(node.position - bridge) < 0.00008) {
                staten_island_nodes.insert(hash);
            }
        }
    }
}


void render_video() {
    vec2 newark_lat_long = vec2(40.694669192970665, -74.18676933576879);
    vec2 zoo_lat_long = vec2(40.767665443249214, -73.97196914550813);

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->config->chill = true;
    gs->label_color = 0xffffffff;
    gs->label_offset = vec2(0, 0.03);
    gs->label_size = vec2(1, 0.065);
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
    unordered_map<double, double> edge_weights;
    if(rendering_on()) {
        load_graph_from_file(g, gs, newark_lat_long, 100000, edge_weights);
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
    gs->config->transition_node_label(MICRO, newark_hash, "\\text{Newark Airport}");
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, zoo_hash, 0xff00ff00);
    gs->config->transition_node_label(MICRO, zoo_hash, "\\text{Central Park Zoo}");
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 2);
    gs->manager.transition(MACRO, "d", ".002");
    gs->render_microblock();
    gs->config->transition_node_label(MICRO, newark_hash, "");
    gs->config->transition_node_label(MICRO, zoo_hash, "");
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
    double increment = 7 * 50. / chunk;
    bool found_goal = false;
    for(int i = 0; i < chunk; i++) {
        if(rendering_on() && !found_goal) found_goal = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }
    for(int i = 0; i < chunk; i++) {
        gs->render_microblock();
    }

    gs->manager.transition(MACRO, "d", ".007");
    stage_macroblock(FileBlock("and then all the twenty minute journeys,"), chunk * 2);
    for(int i = 0; i < chunk; i++) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }
    for(int i = 0; i < chunk; i++) {
        gs->render_microblock();
    }

    bool goal = false;
    stage_macroblock(FileBlock("and so on until it reaches all the forty minute journeys, including the Zoo."), chunk * 2);
    gs->manager.transition(MACRO, "d", ".01");
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !goal) goal = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("The search frontier covers over 65,000 nodes and includes places that are way off,"), 3);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->chill = false;
    gs->render_microblock();

    cout << "Finding nodes in Staten Island..." << endl;
    unordered_set<double> staten_island_nodes;
    if(rendering_on()) get_staten_island_nodes(g, staten_island_nodes);
    cout << "Finding nodes in New Jersey..." << endl;
    unordered_set<double> new_jersey_nodes;
    if(rendering_on()) get_new_jersey_nodes(g, new_jersey_nodes);
    cout << "Staten Island nodes: " << staten_island_nodes.size() << " New Jersey nodes: " << new_jersey_nodes.size() << endl;

    stage_macroblock(FileBlock("like Staten Island,"), 4);
    gs->manager.transition(MACRO, "d", ".004");
    set_camera_to_lat_long(gs, vec2(40.584430, -74.143991), false, MACRO);
    gs->render_microblock();
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 0, edge_weights, staten_island_nodes);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("and large swaths of New Jersey."), 3);
    set_camera_to_lat_long(gs, vec2(40.657, -74.241), false, MICRO);
    gs->manager.transition(MICRO, "d", ".008");
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 0, edge_weights, new_jersey_nodes);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("But even though it searched in illogical directions, the runtime was around 91 milliseconds. Which is incredibly fast."), 4);
    gs->manager.transition(MACRO, "d", ".012");
    gs->render_microblock();
    if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 0, edge_weights, {-1.234567});
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1.5), 1);
    gs->render_microblock();
}
