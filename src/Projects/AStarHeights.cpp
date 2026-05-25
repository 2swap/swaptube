#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

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
        {"globe_opacity", ".7"},
        {"d", ".005"},
        {"texture_or_latlong", ".2"},
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
    // Make a snall list of nodes for which we will draw scaffolding lines propping them up to their new heights, so we can better see the transformation.
    unordered_set<double> scaffold_nodes;
    unordered_set<double> scaffold_bases;
    unordered_map<double, double> scaffold_edges;
    if(rendering_on()) {
        unordered_set<int> random_indices;
        for(int i = 0; i < 50; i++) {
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

    vec4 pos = 0;
    if(rendering_on())
        pos = g->nodes.find(newark_hash)->second.position;
    gs->manager.transition(MICRO, {
        {"theta", "1.55"},
        {"d", ".0085"},
        {"phi", "{t} .25 * sin .22 * 2.14 -"},
        {"x", to_string(pos.x*1.0003)},
        {"y", to_string(pos.y*1.0003)},
        {"z", to_string(pos.z*1.0003)},
    });
    gs->render_microblock();
    heuristic_slide(g, gs, zoo_hash, 1, MICRO, scaffold_bases);
    //fade out scaffold edges
    for(double hash : scaffold_bases) {
        gs->config->fade_edge_color(MICRO, scaffold_edges[hash], hash, opaque_white & 0x00ffffff);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("The penalty A* adds, in this case the distance, is also called a heuristic."), 1);
    gs->render_microblock();
    // Remove all scaffold nodes
    for(double hash : scaffold_bases) {
        g->remove_node(hash);
    }

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
    gs->render_microblock();
    double west_of_newark_hash = get_nearest_node_in_graph(g, vec2(40.713, -74.315));
    g->add_edge(west_of_newark_hash, zoo_hash);
    gs->config->add_edge_if_missing(west_of_newark_hash, zoo_hash);
    gs->config->set_edge_color(west_of_newark_hash, zoo_hash, 0);
    gs->config->transition_edge_color(MICRO, west_of_newark_hash, zoo_hash, 0xff00ff00);
    gs->render_microblock();

    found = false;
    max_dist = 0;
    stage_macroblock(FileBlock("Now the shortest path “illogically” goes west first."), chunks);
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on() && !found) found = run_large_dijkstra(g, gs, newark_hash, zoo_hash, max_dist, 0, edge_weights);
        max_dist += increment;
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("When the heuristic is zero,"), 2);
    gs->config->fade_all_edge_colors(MICRO, opaque_white);
    gs->config->fade_edge_color(MICRO, west_of_newark_hash, zoo_hash, 0xff00ff00);
    heuristic_slide(g, gs, zoo_hash, 0, MICRO);
    gs->render_microblock();
    gs->manager.transition(MICRO, {
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

    stage_macroblock(FileBlock("As we raise the heuristic, our search becomes more and more directed."), chunks);
    if(rendering_on())
        pos = g->nodes.find(newark_hash)->second.position;
    gs->manager.transition(MACRO, {
        {"d", ".008"},
        {"theta", "1.5"},
        {"x", to_string(pos.x*1.0005)},
        {"y", to_string(pos.y*1.0005)},
        {"z", to_string(pos.z*1.0005)},
    });
    heuristic_slide(g, gs, zoo_hash, 1, MACRO);
    unordered_set<double> hack = {1,2};
    float total_microblocks = remaining_microblocks_in_macroblock;
    while(remaining_microblocks_in_macroblock) {
        if(rendering_on()) run_large_dijkstra(g, gs, newark_hash, zoo_hash, 10000, 100000 * (1 - remaining_microblocks_in_macroblock / total_microblocks), edge_weights, hack);
        gs->render_microblock();
    }

    if(rendering_on())
        pos = g->nodes.find(newark_hash)->second.position;
    gs->manager.transition(MACRO, {
        {"x", to_string(pos.x*1.0003)},
        {"y", to_string(pos.y*1.0003)},
        {"z", to_string(pos.z*1.0003)},
    });
    stage_macroblock(FileBlock("At some point, it heads towards Manhattan so aggressively, it doesn't explore west at all and fails to find the direct, faster route."), chunks);
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
