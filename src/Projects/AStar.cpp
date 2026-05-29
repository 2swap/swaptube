#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

// Given lat long coordinates, approximate the distance in kilometers between them using a small angle approximation
int distance_to_target_km(vec2 node_pos, vec2 target_pos) {
    float lat_diff = abs(node_pos.x - target_pos.x);
    float long_diff = abs(node_pos.y - target_pos.y);
    // 111 km per degree of latitude, and 85 km per degree of longitude in NYC
    return sqrt(pow(111 * lat_diff, 2) + pow(85 * long_diff, 2));
}

// For any node whose dot product with the airport is negative (when subtracted by the zoo),
// fade it out over the next microblock, then delete it.
void fade_funnel(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double zoo_hash, double airport_hash) {
    cout << "B" << endl;
    unordered_set<double> to_delete;
    for(auto& [hash, node] : g->nodes) {
        vec4 node_pos = node.position;
        vec4 zoo_to_node = node_pos - g->nodes.find(zoo_hash)->second.position;
        vec4 zoo_to_airport = g->nodes.find(airport_hash)->second.position - g->nodes.find(zoo_hash)->second.position;
        if(dot(zoo_to_node, zoo_to_airport) < 0) {
            to_delete.insert(hash);
            gs->config->fade_node_color(MICRO, hash, 0x00000000);
            unordered_set<double> neighbors = g->get_neighbors(hash);
            for(double neighbor : neighbors) {
                gs->config->fade_edge_color(MICRO, hash, neighbor, opaque_white & 0x00ffffff);
            }
        }
    }
    cout << "C" << endl;
    gs->render_microblock();
    cout << "D" << endl;
    for(double hash : to_delete) {
        g->remove_node(hash);
    }
    cout << "E" << endl;
}
double radiusy = 1000;
void unfade_funnel(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, vec2 newark_lat_long, double zoo_hash, double airport_hash) {
    // Just add nodes back to graph
    bool save = gs->config->chill;
    gs->config->chill = false;
    unordered_map<double, double> dontcare;
    load_graph_from_file(g, gs, newark_lat_long, radiusy, dontcare);
    for(auto& [hash, node] : g->nodes) {
        vec4 node_pos = node.position;
        vec4 zoo_to_node = node_pos - g->nodes.find(zoo_hash)->second.position;
        vec4 zoo_to_airport = g->nodes.find(airport_hash)->second.position - g->nodes.find(zoo_hash)->second.position;
        if(dot(zoo_to_node, zoo_to_airport) < 0) {
            unordered_set<double> neighbors = g->get_neighbors(hash);
            for(double neighbor : neighbors) {
                gs->config->set_edge_color(hash, neighbor, opaque_white & 0x00ffffff);
                gs->config->fade_edge_color(MICRO, hash, neighbor, opaque_white);
            }
        }
    }
    gs->config->set_node_radius(airport_hash, 1);
    gs->config->set_node_radius(zoo_hash, 1);
    gs->config->set_node_color(airport_hash, 0xffff0000);
    gs->config->set_node_color(zoo_hash, 0xff00ff00);
    gs->config->chill = save;
}

void heuristic_slide(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double zoo_hash, double factor, TransitionType tt, unordered_set<double> exclude = {}) {
    for(auto& [hash, node] : g->nodes) {
        if(exclude.count(hash)) continue;
        double distance_to_zoo = length(node.position - g->nodes.find(zoo_hash)->second.position);
        vec4 new_position = normalize(node.position) * (1 + distance_to_zoo * factor / 2);
        gs->transition_node_position(tt, hash, new_position);
    }
}

void render_video() {
    vec2 newark_lat_long = vec2(40.694669192970665, -74.18676933576879);
    vec2 zoo_lat_long = vec2(40.767665443249214, -73.97196914550813);

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"globe_opacity", ".2"},
        {"d", ".005"},
        {"texture_or_latlong", "0"},
    });

    // Fade globe to opacity 1
    vec2 center = newark_lat_long;
    set_camera_to_lat_long(gs, center, true, MICRO);
    stage_macroblock(SilenceBlock(2), 1);
    double newark_hash;
    double zoo_hash;
    unordered_map<double, double> edge_weights;
    if(rendering_on()) {
        load_graph_from_file(g, gs, newark_lat_long, radiusy, edge_weights);
        newark_hash = get_nearest_node_in_graph(g, newark_lat_long);
        zoo_hash = get_nearest_node_in_graph(g, zoo_lat_long);
        gs->config->set_node_radius(newark_hash, 1);
        gs->config->set_node_radius(zoo_hash, 1);
        gs->config->set_node_color(newark_hash, 0xffff0000);
        gs->config->set_node_color(zoo_hash, 0xff00ff00);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("We'd like to prioritize nodes that are closer to the Zoo."), 1);
    gs->manager.transition(MICRO, {
        {"d", ".009"},
    });
    for(auto& [hash, node] : g->nodes) {
        unordered_set<double> neighbors = g->get_neighbors(hash);
        for(double neighbor : neighbors) {
            double distance_to_zoo = length(node.position - g->nodes.find(zoo_hash)->second.position);
            double distance_zoo_to_newark = length(g->nodes.find(zoo_hash)->second.position - g->nodes.find(newark_hash)->second.position);
            double ratio = 0.5 * distance_to_zoo / distance_zoo_to_newark;
            uint32_t color = rainbow(1.7 * sqrt(sqrt(ratio)));
            color = (color & 0x00ffffff) | 0x30000000;
            gs->config->fade_edge_color(MICRO, hash, neighbor, color);
        }
    }
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->render_microblock();

    gs->manager.transition(MICRO, {
        {"globe_opacity", ".5"},
        {"texture_or_latlong", ".5"},
        {"lines_opacity", ".3"},
    });
    stage_macroblock(FileBlock("Using longitudes and latitudes,"), 1);
    gs->render_microblock();

    vector<vec2> lat_longs_to_kms = {
        vec2(40.730610, -73.935242), // 5
        vec2(40.706192, -74.008873), // 7
        vec2(40.660204, -73.968956), // 11
        vec2(40.768, -74.145), // 14
        vec2(40.6618, -74.1213), // 17
        vec2(40.812484, -74.209), // 20
        newark_lat_long + (newark_lat_long - zoo_lat_long) * vec2(.3, .5), // A lot
    };
    stage_macroblock(FileBlock("we can easily calculate the straight line distance between any node and the target."), lat_longs_to_kms.size() * 3);
    int counter = 0;
    int opaque_blue = 0xff88ccff;
    int transparent_blue = 0x0088ccff;
    for (auto& latlong : lat_longs_to_kms) {
        int km = distance_to_target_km(latlong, zoo_lat_long);
        string name = "delete_me_" + counter;
        counter++;
        g->add_node(new HashableString(name));
        double new_hash = HashableString(name).get_hash();
        g->move_node(new_hash, lat_long_to_xyz(latlong));
        g->add_edge(new_hash, zoo_hash);
        gs->config->transition_node_color(MICRO, new_hash, opaque_blue);
        gs->config->set_edge_color(new_hash, zoo_hash, transparent_blue);
        gs->config->transition_edge_color(MICRO, new_hash, zoo_hash, opaque_blue);
        gs->config->transition_edge_label(MICRO, new_hash, zoo_hash, to_string(km) + " \\text{km}");
        gs->render_microblock();

        gs->render_microblock();

        gs->config->fade_edge_color(MICRO, new_hash, zoo_hash, transparent_blue);
        gs->config->transition_edge_label(MICRO, new_hash, zoo_hash, "");
        gs->render_microblock();

        gs->config->set_edge_label(new_hash, zoo_hash, "");
    }

    stage_macroblock(FileBlock("We'll order nodes by their cost plus this straight line distance."), 11);

    // Sort the above 6 nodes by their distance to the zoo
    gs->render_microblock();
    gs->render_microblock();
    counter = 0;
    gs->manager.transition(MACRO, {
        {"globe_opacity", ".2"},
        {"texture_or_latlong", ".2"},
        {"lines_opacity", "1"},
    });
    // transition radius of the above 6 nodes to 2, and label them with "F_a", "F_b", etc (counter 0 is "a", counter 1 is "b", etc)
    for (auto& latlong : lat_longs_to_kms) {
        string name = "delete_me_" + counter;
        counter++;
        double new_hash = HashableString(name).get_hash();
        gs->config->transition_node_radius(MICRO, new_hash, 3);
        string label = "F_" + string(1, 'a' + counter - 1);
        gs->config->transition_node_label(MICRO, new_hash, label);
        gs->render_microblock();
    }
    gs->render_microblock();

    // Simultaneously shrink radii of all the delete me nodes to 0, and remove their labels
    counter = 0;
    for (auto& latlong : lat_longs_to_kms) {
        string name = "delete_me_" + counter;
        counter++;
        double new_hash = HashableString(name).get_hash();
        gs->config->transition_node_radius(MICRO, new_hash, 0);
        gs->config->transition_node_label(MICRO, new_hash, "");
    }
    gs->render_microblock();
    // Delete all the delete me nodes
    counter = 0;
    for (auto& latlong : lat_longs_to_kms) {
        string name = "delete_me_" + counter;
        counter++;
        double new_hash = HashableString(name).get_hash();
        g->remove_node(new_hash);
    }

    // Transition all nodes' positions to scale as a function of their distance to the zoo.
    stage_macroblock(FileBlock("A great way to visualize this comes from the channel Polylog."), 1);
    vec4 pos = 0;
    if(rendering_on())
        pos = g->nodes.find(newark_hash)->second.position;
    gs->manager.transition(MICRO, {
        {"theta", "1.5"},
        {"d", ".0085"},
        {"phi", "{t} .25 * sin .22 * 2.14 -"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("We can represent each node's straight-line distance to the target as a height, so the graph stretches into 3D space."), 2);
    // Make a snall list of nodes for which we will draw scaffolding lines propping them up to their new heights, so we can better see the transformation.
    unordered_set<double> scaffold_nodes;
    unordered_set<double> scaffold_bases;
    unordered_map<double, double> scaffold_edges;
    if(rendering_on()) {
        unordered_set<int> random_indices;
        for(int i = 0; i < 100; i++) {
            int random_index = rand() % g->nodes.size();
            while(random_indices.count(random_index)) {
                random_index = rand() % g->nodes.size();
            }
            random_indices.insert(random_index);
        }
        int index = -1;
        for(auto it = g->nodes.begin(); it != g->nodes.end(); it++) {
            index++;
            if(random_indices.count(index) == 0)
                continue;
            double random_hash = it->first;
            // If the node is on the negative-dot-product side, skip it
            vec4 node_pos = it->second.position;
            vec4 zoo_to_node = node_pos - g->nodes.find(zoo_hash)->second.position;
            vec4 zoo_to_airport = g->nodes.find(newark_hash)->second.position - g->nodes.find(zoo_hash)->second.position;
            if(dot(zoo_to_node, zoo_to_airport) < 0) {
                continue;
            }
            scaffold_nodes.insert(random_hash);
            string new_node_name = "scaffold_" + to_string(index);
            double new_node_hash = HashableString(new_node_name).get_hash();
            scaffold_bases.insert(new_node_hash);
            scaffold_edges[new_node_hash] = random_hash;
            g->add_node(new HashableString(new_node_name));
            g->move_node(new_node_hash, g->nodes.find(random_hash)->second.position);
            gs->config->set_node_color(new_node_hash, 0x00000000);
            g->add_edge(new_node_hash, random_hash);
            gs->config->add_edge_if_missing(new_node_hash, random_hash);
            gs->config->set_edge_color(new_node_hash, random_hash, opaque_white);
        }
    }

    gs->render_microblock();
    heuristic_slide(g, gs, zoo_hash, 1, MICRO, scaffold_bases);
    gs->manager.transition(MICRO, {
        {"x", to_string(pos.x*1.0005)},
        {"y", to_string(pos.y*1.0005)},
        {"z", to_string(pos.z*1.0005)},
    });
    cout << "A" << endl;
    fade_funnel(g, gs, zoo_hash, newark_hash);

    stage_macroblock(FileBlock("The further a node is from the Zoo, the higher it is."), 1);
    //fade out scaffold edges
    for(double hash : scaffold_bases) {
        gs->config->fade_edge_color(MICRO, scaffold_edges[hash], hash, opaque_white & 0x00ffffff);
    }
    gs->render_microblock();
    // Remove all scaffold nodes
    for(double hash : scaffold_bases) {
        g->remove_node(hash);
    }

    int chunks = 100;
    gs->config->chill = true;
    stage_macroblock(FileBlock("The algorithm penalizes moves that climb up the graph"), chunks);
    double max_dist = 0;
    bool found = false;
    double increment = 16 * 100. / chunks;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, -200000, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(SilenceBlock(.5), 1);
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->render_microblock();

    stage_macroblock(FileBlock("instead of funnel down it."), chunks);
    max_dist = 0;
    found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 150000, edge_weights);
        max_dist += increment * .7;
        gs->render_microblock();
    }

    stage_macroblock(SilenceBlock(.5), 1);
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->render_microblock();

    gs->config->chill = false;
    stage_macroblock(FileBlock("That way, nodes in the opposite direction won't be explored early on."), 7);
    gs->render_microblock();
    for(int i = 0; i < 3; i++) {
        for(auto& [hash, node] : g->nodes) {
            unordered_set<double> neighbors = g->get_neighbors(hash);
            for(double neighbor : neighbors) {
                double distance_to_zoo = length(node.position - g->nodes.find(zoo_hash)->second.position);
                double distance_zoo_to_newark = length(g->nodes.find(zoo_hash)->second.position - g->nodes.find(newark_hash)->second.position);
                double extra_distance = 2 * (distance_to_zoo - 1.3*distance_zoo_to_newark) / distance_zoo_to_newark;
                extra_distance = max(extra_distance, 0.0);
                extra_distance = min(extra_distance, 1.0);
                uint32_t color = colorlerp(opaque_white, 0x30ff0000, extra_distance);
                gs->config->set_edge_color(hash, neighbor, color);
            }
        }
        gs->render_microblock();
        gs->config->set_all_edge_colors(opaque_white);
        gs->render_microblock();
    }

    gs->config->chill = true;
    stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("So while Dijkstra's search frontier")), 2);
    gs->manager.transition(MACRO, {
        {"d", ".005"},
    });
    gs->manager.transition(MACRO, {
        {"phi", "<timey> .11 * sin .2 *"},
        {"theta", ".5"},
    });
    heuristic_slide(g, gs, zoo_hash, 0, MICRO);
    gs->render_microblock();
    if(rendering_on()) unfade_funnel(g, gs, newark_lat_long, zoo_hash, newark_hash);
    gs->render_microblock();

    stage_macroblock(FileBlock("spreads out in all directions,"), chunks);
    gs->manager.transition(MACRO, {
        {"d", ".009"},
    });
    set_camera_to_lat_long(gs, newark_lat_long, false, MACRO);
    max_dist = 0;
    found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("this modified Dijkstra's, also called A*,"), 1);
    gs->manager.transition(MICRO, {
        {"d", ".003"},
    });
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->render_microblock();

    stage_macroblock(FileBlock("immediately heads towards the Zoo."), chunks);
    gs->manager.transition(MACRO, {
        {"d", ".006"},
    });
    max_dist = 0;
    found = false;
    center = (newark_lat_long + zoo_lat_long) / 2.f;
    set_camera_to_lat_long(gs, center, false, MACRO);
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 300000, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("It only checks around 7,000 nodes — that’s almost a 10x improvement from Dijkstra’s!"), 1);
    gs->manager.transition(MICRO, {
        {"theta", "1.05"},
        {"d", ".0085"},
        {"phi", "{t} .25 * sin .16 * 2.14 -"},
        {"x", to_string(pos.x*1.0003)},
        {"y", to_string(pos.y*1.0003)},
        {"z", to_string(pos.z*1.0003)},
    });
    heuristic_slide(g, gs, zoo_hash, 1, MACRO);
    fade_funnel(g, gs, zoo_hash, newark_hash);

    stage_macroblock(FileBlock("And by changing this penalty, also called a heuristic, we can tweak how aggressively A* targets the Zoo. "), chunks);
    if(rendering_on())
        pos = g->nodes.find(newark_hash)->second.position;
    gs->manager.transition(MACRO, {
        {"d", ".0105"},
    });
    heuristic_slide(g, gs, zoo_hash, .5, MACRO);
    float total_microblocks = remaining_microblocks_in_macroblock;
    unordered_set<double> hack = {1,2};
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 300000 - 150000 * (1 - remaining_microblocks_in_macroblock / total_microblocks), edge_weights, hack);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("The steeper the slope, the less A* spreads out."), chunks);
    if(rendering_on())
        pos = g->nodes.find(newark_hash)->second.position;
    gs->manager.transition(MACRO, {
        {"theta", "1.4"},
        {"x", to_string(pos.x*1.001)},
        {"y", to_string(pos.y*1.001)},
        {"z", to_string(pos.z*1.001)},
        {"phi", "{t} .25 * sin .06 * 2.14 -"},
    });
    heuristic_slide(g, gs, zoo_hash, 2, MACRO);
    total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 150000 + 400000 * (1 - remaining_microblocks_in_macroblock / total_microblocks), edge_weights, hack);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("It's a more human approach to finding the shortest path since we"), 1);
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    heuristic_slide(g, gs, zoo_hash, 1, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("only explore in the rough direction of the target."), chunks);
    gs->manager.transition(MACRO, {
        {"theta", "1.4"},
        {"x", to_string(pos.x*1.001)},
        {"y", to_string(pos.y*1.001)},
        {"z", to_string(pos.z*1.001)},
        {"phi", "{t} .25 * sin .06 * 2.14 -"},
    });
    total_microblocks = remaining_microblocks_in_macroblock;
    max_dist = 0;
    found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 150000, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();
}
