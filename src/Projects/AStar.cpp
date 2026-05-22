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
        load_graph_from_file(g, gs, newark_lat_long, 21, edge_weights);
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

    stage_macroblock(FileBlock("We'll order nodes by their cost plus this straight line distance."), 7);
    // Sort the above 6 nodes by their distance to the zoo
    counter = 0;
    gs->manager.transition(MACRO, {
        {"globe_opacity", ".2"},
        {"texture_or_latlong", "0"},
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

    stage_macroblock(FileBlock("Nodes in the opposite direction won't be explored early on."), 7);
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

    // Re-do Dijkstra's
    stage_macroblock(SilenceBlock(1), 1);
    gs->manager.transition(MICRO, {
        {"d", ".004"},
    });
    gs->render_microblock();

    gs->config->chill = true;
    int chunks = 100;
    stage_macroblock(FileBlock("With Dijkstra’s, the search frontier spreads out in all directions."), chunks);
    gs->manager.transition(MACRO, {
        {"d", ".008"},
    });
    float max_dist = 0;
    float increment = 16 * 100. / chunks;
    bool found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("But this modified Dijkstra's"), 1);
    gs->manager.transition(MICRO, {
        {"d", ".003"},
    });
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->render_microblock();

    stage_macroblock(FileBlock("also called A* immediately heads towards the zoo."), chunks);
    gs->manager.transition(MACRO, {
        {"d", ".006"},
    });
    center = (newark_lat_long + zoo_lat_long) / 2.f;
    set_camera_to_lat_long(gs, center, false, MACRO);
    max_dist = 0;
    found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 300000, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("It only checks around 7,000 nodes — that’s almost a 10x improvement!"), 1);
    gs->manager.transition(MACRO, {
        {"d", ".008"},
    });
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1.5), 1);
    gs->render_microblock();
}
