#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

void heuristic_slide(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double zoo_hash, double factor, TransitionType tt) {
    for(auto& [hash, node] : g->nodes) {
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
        {"globe_opacity", ".5"},
        {"d", ".005"},
        {"texture_or_latlong", "0"},
    });
    gs->manager.set({
        {"theta", ".9"},
    });
    gs->config->chill = true;

    // Fade globe to opacity 1
    vec2 center = newark_lat_long;
    set_camera_to_lat_long(gs, center, true, MICRO, 1.0003);
    stage_macroblock(SilenceBlock(2), 1);
    double newark_hash;
    double zoo_hash;
    unordered_map<double, double> edge_weights;
    if(rendering_on()) {
        load_graph_from_file(g, gs, newark_lat_long, 0.21, edge_weights);
        newark_hash = get_nearest_node_in_graph(g, newark_lat_long);
        zoo_hash = get_nearest_node_in_graph(g, zoo_lat_long);
        gs->config->set_node_radius(newark_hash, 1);
        gs->config->set_node_radius(zoo_hash, 1);
        gs->config->set_node_color(newark_hash, 0xffff0000);
        gs->config->set_node_color(zoo_hash, 0xff00ff00);
    }
    gs->render_microblock();

    // Transition all nodes' positions to scale as a function of their distance to the zoo.
    stage_macroblock(FileBlock("Each node's height is its straight line distance to the target."), 2);
    gs->manager.transition(MICRO, {
        {"theta", "1.55"},
        {"d", ".01"},
        {"phi", "{t} .3 * sin .2 * 2.14 -"},
    });
    gs->render_microblock();
    heuristic_slide(g, gs, zoo_hash, 1, MICRO);
    gs->render_microblock();

    stage_macroblock(FileBlock("The penalty A* adds, in this case the distance, is also called a heuristic."), 1);
    gs->render_microblock();

    int chunks = 100;
    stage_macroblock(FileBlock("Now we can really see how the heuristic funnels the search directly towards Central Park."), chunks);
    double max_dist = 0;
    float increment = 16 * 100. / chunks;
    bool found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 300000, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("But say there’s a direct route to Central Park that starts just west of Newark."), 3);
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->render_microblock();
    double west_of_newark_hash = get_nearest_node_in_graph(g, vec2(40.713, -74.315));
    g->add_edge(west_of_newark_hash, zoo_hash);
    gs->config->add_edge_if_missing(west_of_newark_hash, zoo_hash);
    gs->config->set_edge_color(west_of_newark_hash, zoo_hash, 0);
    gs->config->transition_edge_color(MICRO, west_of_newark_hash, zoo_hash, 0xff00ff00);
    gs->render_microblock();
    gs->render_microblock();

    found = false;
    stage_macroblock(FileBlock("Now the shortest path “illogically” goes west first."), chunks);
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("When the heuristic is zero,"), 2);
    heuristic_slide(g, gs, zoo_hash, 0, MICRO);
    gs->render_microblock();
    gs->manager.transition(MACRO, {
        {"theta", "1"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("A* is the same as Dijkstra’s,"), chunks / 2);
    max_dist = 0;
    found = false;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("so it still finds the shortest path."), 1);
    gs->manager.transition(MICRO, {
        {"theta", "1.4"},
    });
    gs->render_microblock();

    gs->config->chill = false;
    stage_macroblock(FileBlock("As we raise the heuristic, our search becomes more and more directed."), chunks);
    set_camera_to_lat_long(gs, center, true, MICRO, 1.0006);
    gs->manager.transition(MACRO, {
        {"d", ".009"},
        {"theta", "1.5"},
    });
    heuristic_slide(g, gs, zoo_hash, 1, MACRO);
    unordered_set<double> hack = {1,2};
    float total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 100000 * (1 - remaining_microblocks_in_macroblock / total_microblocks), edge_weights, hack);
        gs->render_microblock();
    }

    set_camera_to_lat_long(gs, center, true, MICRO, 1.0009);
    stage_macroblock(FileBlock("At some point, it heads towards Manhattan so aggressively, it doesn't explore west at all and fails to find the direct route."), chunks);
    heuristic_slide(g, gs, zoo_hash, 2, MACRO);
    total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 100000 + 100000 * (1 - remaining_microblocks_in_macroblock / total_microblocks), edge_weights, hack);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("We don’t want to raise the heuristic so much that it overestimates the actual path lengths."), chunks);
    heuristic_slide(g, gs, zoo_hash, 1, MACRO);
    total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 200000 - 100000 * (1 - remaining_microblocks_in_macroblock / total_microblocks), edge_weights, hack);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("It should always be an underestimate."), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();
}
