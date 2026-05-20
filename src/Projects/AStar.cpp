#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

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

    unordered_map<int, vec2> lat_longs_to_kms = {
        // Kearny
        //{17, vec2(40.768, -74.145)},
        // Montclair
        {22, vec2(40.812484, -74.209)},
        // Prospect Park
        {12, vec2(40.660204, -73.968956)},
    };
    stage_macroblock(FileBlock("we can easily calculate the straight line distance between any node and the target."), lat_longs_to_kms.size() * 3);
    for (auto& [km, latlong] : lat_longs_to_kms) {
        g->add_node(new HashableString("delete_me"));
        double new_hash = HashableString("delete_me").get_hash();
        g->move_node(new_hash, lat_long_to_xyz(latlong));
        g->add_edge(new_hash, zoo_hash);
        gs->config->transition_node_color(MICRO, new_hash, 0xffff0000);
        gs->config->set_edge_color(new_hash, zoo_hash, 0x00ff0000);
        gs->config->transition_edge_color(MICRO, new_hash, zoo_hash, 0xffff0000);
        gs->config->transition_edge_label(MICRO, new_hash, zoo_hash, to_string(km) + " \\text{km}");
        gs->render_microblock();

        gs->render_microblock();

        gs->config->fade_edge_color(MICRO, new_hash, zoo_hash, 0x00ff0000);
        gs->config->transition_edge_label(MICRO, new_hash, zoo_hash, "");
        gs->config->fade_node_color(MICRO, new_hash, 0x00ff0000);
        gs->render_microblock();

        g->remove_node(new_hash);
        gs->config->set_edge_label(new_hash, zoo_hash, "");
    }

    stage_macroblock(FileBlock("We'll order nodes by their cost plus this straight line distance."), 1);
    gs->manager.transition(MICRO, {
        {"globe_opacity", ".2"},
        {"texture_or_latlong", "0"},
        {"lines_opacity", "1"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("Nodes in the opposite direction won't be explored early on."), 6);
    for(int i = 0; i < 3; i++) {
        for(auto& [hash, node] : g->nodes) {
            unordered_set<double> neighbors = g->get_neighbors(hash);
            for(double neighbor : neighbors) {
                double distance_to_zoo = length(node.position - g->nodes.find(zoo_hash)->second.position);
                double distance_zoo_to_newark = length(g->nodes.find(zoo_hash)->second.position - g->nodes.find(newark_hash)->second.position);
                double extra_distance = 2 * (distance_to_zoo - distance_zoo_to_newark) / distance_zoo_to_newark;
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
        {"d", ".008"},
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
    gs->render_microblock();
}
