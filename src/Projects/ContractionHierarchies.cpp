#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

unordered_map<string, vec2> graph_nodes = {
    {"a", vec2(-1.5, 0)},
    {"b", vec2(-.5, .5)},
    {"c", vec2(.5, .5)},
    {"d", vec2(1.5, .5)},
    {"e", vec2(-.5, -.5)},
    {"f", vec2(.5, -.5)},
    {"h", vec2(1.5, -.5)},
};

unordered_map<string, int> graph_edges_with_weights = {
    {"ab", 2},
    {"bc", 3},
    {"cd", 1},
    {"ae", 10},
    {"ef", 4},
    {"fh", 7},
    {"ec", 5},
    {"cf", 3},
    {"fd", 1},
};

void step1() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);

    gs->manager.set({
        {"q1", "1"},
        {"qi", "{t} .09 * sin .03 *"},
        {"qj", "{t} .07 * cos .03 *"},
        {"qk", "0"},
        {"d", "4"},
        {"edge_weights_size", "1.2"},
        {"points_radius_multiplier", "2.2"},
    });

    stage_macroblock(FileBlock("Let's zoom in on a smaller graph to see how this really works."), 1);
    for(auto& [node, coords] : graph_nodes) {
        double hash = HashableString(node).get_hash();
        vec4 position = vec4(coords.x, coords.y, 0, 1);
        g->add_node(new HashableString(node));
        g->move_node(hash, position);
    }
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        g->add_edge(hash1, hash2);
        gs->config->set_edge_label(hash1, hash2, to_string(weight));
    }
    gs->render_microblock();

    gs->manager.transition(MICRO, "edge_weights_size", "0");
    stage_macroblock(FileBlock("We ignore the edge weights for the first step."), 1);
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, "");
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("These two nodes split the graph roughly in half,"), 7);
    // Highlight c and e
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("e").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->fade_edge_color(MICRO, HashableString("c").get_hash(), HashableString("e").get_hash(), 0xffff0000);
    gs->render_microblock();
    int split_edge_color = 0x10ffffff;
    gs->config->transition_edge_color(MICRO, HashableString("c").get_hash(), HashableString("d").get_hash(), split_edge_color);
    gs->config->transition_edge_color(MICRO, HashableString("c").get_hash(), HashableString("b").get_hash(), split_edge_color);
    gs->config->transition_edge_color(MICRO, HashableString("e").get_hash(), HashableString("a").get_hash(), split_edge_color);
    gs->config->transition_edge_color(MICRO, HashableString("e").get_hash(), HashableString("f").get_hash(), split_edge_color);
    gs->config->transition_edge_color(MICRO, HashableString("c").get_hash(), HashableString("f").get_hash(), split_edge_color);
    gs->render_microblock();
    // Make left side green
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xff00ff00);
    gs->config->transition_node_color(MICRO, HashableString("b").get_hash(), 0xff00ff00);
    gs->config->set_edge_color(HashableString("a").get_hash(), HashableString("b").get_hash(), 0xff00ff00);
    gs->render_microblock();
    // Make right side blue
    gs->config->transition_node_color(MICRO, HashableString("d").get_hash(), 0xff0088ff);
    gs->config->transition_node_color(MICRO, HashableString("f").get_hash(), 0xff0088ff);
    gs->config->transition_node_color(MICRO, HashableString("h").get_hash(), 0xff0088ff);
    gs->config->set_edge_color(HashableString("f").get_hash(), HashableString("d").get_hash(), 0xff0088ff);
    gs->config->set_edge_color(HashableString("h").get_hash(), HashableString("f").get_hash(), 0xff0088ff);
    gs->render_microblock();

    unordered_map<string, int> node_ranks = {
        {"a", 2},
        {"b", 1},
        {"c", 6},
        {"d", 4},
        {"e", 7},
        {"f", 5},
        {"h", 3},
    };

    stage_macroblock(FileBlock("so we give them the highest rank."), 2);
    gs->render_microblock();
    gs->config->transition_node_label(MICRO, HashableString("c").get_hash(), to_string(node_ranks["c"]));
    gs->config->transition_node_label(MICRO, HashableString("e").get_hash(), to_string(node_ranks["e"]));
    gs->render_microblock();

    // Fade all nodes and edges gray except for d, f and h
    stage_macroblock(FileBlock("We can split the right half a little further,"), 2);
    gs->render_microblock();

    // Highlight f
    gs->config->transition_node_color(MICRO, HashableString("f").get_hash(), 0xff00ff00);
    gs->config->transition_node_label(MICRO, HashableString("f").get_hash(), to_string(node_ranks["f"]));
    gs->config->transition_edge_color(MICRO, HashableString("f").get_hash(), HashableString("d").get_hash(), split_edge_color);
    gs->config->transition_edge_color(MICRO, HashableString("f").get_hash(), HashableString("h").get_hash(), split_edge_color);
    gs->render_microblock();

    stage_macroblock(FileBlock("so we give that node the next highest rank."), 2);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("We then rank the rest of the nodes in any order with the remaining numbers."), 6);
    gs->render_microblock();
    string remaining_nodes = "abdh";
    double height = 0;
    for(char node : remaining_nodes) {
        string node_str(1, node);
        vec2 position = graph_nodes[node_str];
        gs->config->transition_node_label(MICRO, HashableString(node_str).get_hash(), to_string(node_ranks[node_str]));
        gs->config->splash_node(HashableString(node_str).get_hash());
        gs->render_microblock();
        height -= .5;
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("We can show this ranking graphically."), 3);
    // Duplicate the whole graph
    for(auto& [node, coords] : graph_nodes) {
        double hash = HashableString(node).get_hash();
        vec4 position = vec4(coords.x, coords.y, 0, 1);
        g->add_node(new HashableString(node + "'"));
        gs->config->add_node_if_missing(HashableString(node + "'").get_hash());
        gs->config->set_node_color(HashableString(node + "'").get_hash(), 0xffffffff);
        gs->config->set_node_label(HashableString(node + "'").get_hash(), to_string(node_ranks[node]));
        g->move_node(HashableString(node + "'").get_hash(), position);
    }
    // Add edges
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        g->add_edge(HashableString(node1 + "'").get_hash(), HashableString(node2 + "'").get_hash());
        gs->config->set_edge_color(HashableString(node1 + "'").get_hash(), HashableString(node2 + "'").get_hash(), 0xffffffff);
    }
    gs->config->set_all_edge_colors(0xffffffff);
    gs->render_microblock();

    for(auto& [node, coords] : graph_nodes) {
        string node_str = node;
        vec2 position = graph_nodes[node_str];
        gs->transition_node_position(MICRO, HashableString(node_str).get_hash(), vec4(position.x, position.y + 1.8, 0, 0));
        gs->transition_node_position(MICRO, HashableString(node_str + "'").get_hash(), vec4(position.x, position.y + 1.8, 0, 0));
    }
    gs->manager.transition(MICRO, "d", "7");
    gs->render_microblock();

    for(auto& [node, coords] : graph_nodes) {
        string node_str = node;
        vec2 position = graph_nodes[node_str];
        gs->transition_node_position(MICRO, HashableString(node_str + "'").get_hash(), vec4(position.x, position.y + 1.2, 0, 0));
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("We pull nodes down, or contract them, to show their rank."), 1);
    gs->render_microblock();

    unordered_map<string, vec2> contracted_positions = {
        {"a", vec2(-1.5, 2)},
        {"b", vec2(.7, 1)},
        {"c", vec2(.2, 6)},
        {"d", vec2(1, 4)},
        {"e", vec2(.2, 8)},
        {"f", vec2(2, 5)},
        {"h", vec2(2, 3)},

        {"i", vec2(-1, -1)},
        {"j", vec2(0, 0)},

        {"k", vec2(-1, -1)},
        {"l", vec2(0, 0)},
    };

    stage_macroblock(CompositeBlock(FileBlock("1 is the furthest down, 2 is next up and so on."), SilenceBlock(3)), 10);
    for (int rank = 1; rank <= 7; rank++) {
        // Find node with this rank
        string node;
        for(auto& [n, r] : node_ranks) {
            if(r == rank) {
                node = n;
                break;
            }
        }
        vec2 position = graph_nodes[node];
        string prime = node + "'";
        gs->transition_node_position(MICRO, HashableString(prime).get_hash(), vec4(contracted_positions[node].x, contracted_positions[node].y / 3. - 2, 0, 0));
        if(node == "b" || node == "a" || node == "h") {
            gs->render_microblock();
        }
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("Eventually, our algorithm will use a bidirectional search starting from the source and target,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("but on both sides, it will only search upwards from low ranks to higher ranks."), 1);
    // Transition all edge colors from side of low rank to side of high rank
    for (auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        string lower_rank_node = node_ranks[node1] < node_ranks[node2] ? node1 : node2;
        string higher_rank_node = node_ranks[node1] < node_ranks[node2] ? node2 : node1;
        double hash1 = HashableString(lower_rank_node + "'").get_hash();
        double hash2 = HashableString(higher_rank_node + "'").get_hash();
        gs->config->transition_edge_color(MICRO, hash1, hash2, 0xffff0000);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("But if we search like that, there's a few problems."), 1);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Let’s add the edge weights back."), 1);
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(weight));
        hash1 = HashableString(node1 + "'").get_hash();
        hash2 = HashableString(node2 + "'").get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(weight));
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("What’s the shortest path from node 2 to node 6 using this hierarchy search?"), 5);
    // Highlight A and C
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xffff0000);
    gs->config->transition_node_color(MICRO, HashableString("a'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xff00ff00);
    gs->config->transition_node_color(MICRO, HashableString("c'").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Well, starting from A, we search up and reach E with a cost of 10."), 4);
    gs->render_microblock();
    // Splash A
    gs->config->splash_node(HashableString("a'").get_hash());
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("e'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("e'").get_hash(), 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("From C, we also search up and reach E with a cost of 5."), 2);
    // Add edge C to E
    gs->config->transition_edge_color(MICRO, HashableString("c'").get_hash(), HashableString("e'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->splash_node(HashableString("e'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Shortest path is A, E, C with a cost of 15. Done."), 15);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "e'", "c'"}, 0xffff0001);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Except that’s not the shortest path."), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Using the path A, B, C only costs 5."), 15);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("What went wrong?"), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("B has edges to two higher ranked nodes."), 4);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("b'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, HashableString("b'").get_hash(), HashableString("a'").get_hash(), 0xffff0000);
    gs->config->transition_edge_color(MICRO, HashableString("b'").get_hash(), HashableString("c'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("a'").get_hash(), 0xffff0000);
    gs->config->transition_node_color(MICRO, HashableString("c'").get_hash(), 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("But since it’s the lowest of the three, there’s no way to consider that path."), 3);
    // Pop B
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Remember, we only ever search upwards from the source and the target."), 13);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    // Highlight path A, B, C
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);

    stage_macroblock(FileBlock("To fix it, we’ll add a shortcut between A and C."), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    g->add_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());
    gs->config->set_edge_dashed(HashableString("a'").get_hash(), HashableString("c'").get_hash(), true);
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x2000ffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Its weight is just the cost of the path A, B, C,"), 5);
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "\\text{5 (via b)}");
    gs->render_microblock();

    stage_macroblock(FileBlock("The bidirectional search now checks both the upper path and the shortcut to find the best path."), 12);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "e'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"a'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("and it has a midpoint marker to B."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But there's three more problems if we want this to work on ANY graph."), 1);
    // Fade out shortcut and then remove
    gs->config->fade_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "");
    gs->render_microblock();
    g->remove_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());

    stage_macroblock(FileBlock("First, on a slightly larger graph,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("it's possible this node wasn’t the only lower ranked node connected up to these two nodes."), 11);
    gs->render_microblock();
    // Splash node b
    gs->config->splash_node(HashableString("b'").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"b'", "a'"}, 0xffff0001);
    gs->render_microblock();
    trace_path(gs, {"b'", "c'"}, 0xffff0001);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("There could be several lower nodes."), 4);
    // Add nodes i and j, each connected to c and a
    gs->render_microblock();
    gs->render_microblock();
    g->add_node(new HashableString("i'"));
    g->move_node(HashableString("i'").get_hash(), vec4(contracted_positions["i"].x, contracted_positions["i"].y / 3. - 2, 0, 0));
    g->add_edge(HashableString("i'").get_hash(), HashableString("a'").get_hash());
    g->add_edge(HashableString("i'").get_hash(), HashableString("c'").get_hash());
    gs->render_microblock();
    g->add_node(new HashableString("j'"));
    g->move_node(HashableString("j'").get_hash(), vec4(contracted_positions["j"].x, contracted_positions["j"].y / 3. - 2, 0, 0));
    g->add_edge(HashableString("j'").get_hash(), HashableString("a'").get_hash());
    g->add_edge(HashableString("j'").get_hash(), HashableString("c'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Any one of these paths, also called lower triangles, could be the shortest."), 15);
    // Transition path a' b' c'
    gs->render_microblock();
    trace_path(gs, {"a'", "i'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    // transition path a' i' c'
    gs->render_microblock();
    trace_path(gs, {"a'", "j'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    // transition path a' j' c'
    gs->render_microblock();
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("So in general, the shortcut is the minimum cost of all of these lower triangles."), 2);
    gs->config->fade_node_color(MICRO, HashableString("i'").get_hash(), 0x00000000);
    gs->config->fade_node_color(MICRO, HashableString("j'").get_hash(), 0x00000000);
    // Fade out all edges connected to i and j
    gs->config->fade_edge_color(MICRO, HashableString("i'").get_hash(), HashableString("a'").get_hash(), 0x00000000);
    gs->config->fade_edge_color(MICRO, HashableString("i'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->fade_edge_color(MICRO, HashableString("j'").get_hash(), HashableString("a'").get_hash(), 0x00000000);
    gs->config->fade_edge_color(MICRO, HashableString("j'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->render_microblock();
    g->remove_node(HashableString("i'").get_hash());
    g->remove_node(HashableString("j'").get_hash());
    // fade out shortcut
    gs->config->fade_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->render_microblock();
    g->remove_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());

    stage_macroblock(FileBlock("Second, it’s possible the shortest path between these two nodes"), 5);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("a'").get_hash());
    gs->config->splash_node(HashableString("c'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("goes through multiple lower ranked nodes."), 5);
    gs->render_microblock();
    // Add node k and l
    g->add_node(new HashableString("k'"));
    g->add_node(new HashableString("l'"));
    g->move_node(HashableString("k'").get_hash(), vec4(contracted_positions["k"].x, contracted_positions["k"].y / 3. - 2, 0, 0));
    g->move_node(HashableString("l'").get_hash(), vec4(contracted_positions["l"].x, contracted_positions["l"].y / 3. - 2, 0, 0));
    gs->config->add_node_if_missing(HashableString("k'").get_hash());
    gs->config->add_node_if_missing(HashableString("l'").get_hash());
    gs->config->set_node_color(HashableString("k'").get_hash(), 0x00ffffff);
    gs->config->set_node_color(HashableString("l'").get_hash(), 0x00ffffff);

    // Connect a-k-l-c
    g->add_edge(HashableString("a'").get_hash(), HashableString("k'").get_hash());
    g->add_edge(HashableString("k'").get_hash(), HashableString("l'").get_hash());
    g->add_edge(HashableString("l'").get_hash(), HashableString("c'").get_hash());
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("k'").get_hash(), "2");
    gs->config->transition_edge_label(MICRO, HashableString("k'").get_hash(), HashableString("l'").get_hash(), "2");
    gs->config->transition_edge_label(MICRO, HashableString("l'").get_hash(), HashableString("c'").get_hash(), "2");
    trace_path(gs, {"a'", "k'", "l'", "c'"}, 0xffff0000);

    stage_macroblock(FileBlock("The whole path isn't a lower triangle."), 10);
    trace_path(gs, {"a'", "k'", "l'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Here's what we'll do:"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Starting from the lowest node,"), 2);
    gs->render_microblock();
    gs->config->splash_node(HashableString("k'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("check to see if it connects to two higher nodes."), 3);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, HashableString("k'").get_hash(), HashableString("a'").get_hash(), 0xfffffffe);
    gs->config->transition_edge_color(MICRO, HashableString("k'").get_hash(), HashableString("l'").get_hash(), 0xfffffffe);
    gs->render_microblock();

    stage_macroblock(FileBlock("If not, move on to the next node."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("If yes, then add a shortcut,"), 2);
    gs->render_microblock();
    g->add_edge(HashableString("a'").get_hash(), HashableString("l'").get_hash());
    gs->config->set_edge_dashed(HashableString("a'").get_hash(), HashableString("l'").get_hash(), true);
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("l'").get_hash(), 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("l'").get_hash(), 0x2000ffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("and just like before, its weight is the minimum cost over its lower triangles."), 7);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("l'").get_hash(), "\\text{2 (via k)}");
    trace_path(gs, {"a'", "k'", "l'"}, 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("Move on to the next lowest node, and repeat."), 2);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    // Splash l'
    gs->config->splash_node(HashableString("l'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Now node 2 forms a lower triangle!"), 3);
    trace_path(gs, {"a'", "l'", "c'"}, 0xffff0000);

    stage_macroblock(FileBlock("We add a shortcut and compare the two lower triangles to get the minimum cost."), 10);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    g->add_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());
    gs->config->set_edge_dashed(HashableString("a'").get_hash(), HashableString("c'").get_hash(), true);
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x2000ffff);
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "l'", "c'"}, 0xffff0000);
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "\\text{5 (via b)}");
    gs->render_microblock();

    stage_macroblock(FileBlock("Finally, the shortcut between these two nodes points to the right path."), 8);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    // splash a and c
    gs->config->splash_node(HashableString("a'").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("c'").get_hash());
    gs->render_microblock();
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("A path that only uses lower ranked nodes is like a path that only uses small, local roads."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("We don’t want the algorithm to return the wrong path if a local route really is better than taking the highway."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Building shortcuts from the bottom up makes sure we don’t miss these paths."), 10);
    //trace_path(gs, {"a'", "k'", "l'"}, 0xffff0000);
    trace_path(gs, {"a'", "l'"}, 0xff0080ff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    //trace_path(gs, {"a'", "l'", "c'"}, 0xffff0000);
    trace_path(gs, {"a'", "c'"}, 0xff0080ff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("After all, if local roads like these make up the shortest path, then they will have the minimum cost."), 14);
    // Change weights of a-k, k-l, and l-c all to 1.
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("k'").get_hash(), "1");
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("k'").get_hash(), HashableString("l'").get_hash(), "1");
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("l'").get_hash(), HashableString("c'").get_hash(), "1");
    trace_path(gs, {"a'", "k'", "l'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "k'", "l'", "c'"}, 0xffffffff);

    stage_macroblock(FileBlock("And as we build shortcuts, the minimum cost ‘bubbles up’."), 3);
    // Change a-l shortcut to weight 2
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("l'").get_hash(), "\\text{2 (via k)}");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Eventually, it moves up high enough that our search will check it."), 3);
    // change a-c shortcut to weight 3
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "\\text{3 (via l)}");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("If a lower path _isn’t_ the minimum, then the shortcuts won’t point back to it."), 7);
    // abc is no longer the minimum, highlight it
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("We don’t need to check it again."), 4);
    trace_path(gs, {"a'", "b'", "c'"}, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("This way, we both limit the search space and never miss the shortest path."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("For the final problem:"), 1);
    gs->config->fade_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    g->remove_node(HashableString("k'").get_hash());
    g->remove_node(HashableString("l'").get_hash());
    gs->render_microblock();
    g->remove_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());

    stage_macroblock(FileBlock("What’s the shortest path from this node to this one?"), 4);
    // splash c' and f'
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("c'").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("f'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Before, we had to add a shortcut because the edge didn’t exist."), 5);
    gs->render_microblock();
    trace_path(gs, {"a'", "c'"}, 0xfffffffe);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Here, an edge already exists where we’d put a shortcut for this lower triangle."), 7);
    // f d c is a lower triangle, even though edge c-f exists
    trace_path(gs, {"c'", "f'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"f'", "d'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("We need to run the same bottom up check for any lower triangle."), 1);
    gs->config->fade_edge_color(MICRO, HashableString("f'").get_hash(), HashableString("c'").get_hash(), 0x00ffffff);
    gs->config->transition_edge_label(MICRO, HashableString("f'").get_hash(), HashableString("c'").get_hash(), "\\text{2 (via d)}");
    gs->render_microblock();

    stage_macroblock(FileBlock("Then we replace the original edges with shortcuts and set the weight to be minimum cost over the lower triangles."), 2);
    trace_path(gs, {"c'", "f'"}, 0xff00ff00);

    stage_macroblock(FileBlock("Now the graph is ready!"), 1);
    gs->render_microblock();
}

void bad() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "14"},
        {"edge_weights_size", "1"},
        {"points_radius_multiplier", "3"},
    });

    vec2 left_graph_delta = vec2(-10, -10);
    vec2 right_graph_delta = vec2(-10, 10);

    unordered_map<string, int> left_node_ranks = {
        {"a", 1},
        {"b", 2},
        {"c", 3},
        {"d", 4},
        {"e", 5},
        {"f", 6},
        {"g", 7},
        {"h", 8},
        {"i", 9},
    };
    unordered_map<string, int> right_node_ranks = {
        {"a", 8},
        {"b", 7},
        {"c", 6},
        {"d", 9},
        {"e", 1},
        {"f", 2},
        {"g", 3},
        {"h", 4},
        {"i", 5},
    };

    stage_macroblock(FileBlock("For a linear sequence of nodes,"), 9);
    for(int i = 0; i < 9; i++) {
        for (char suffix : {'l', 'c', 'r'}) {
            string node_name = string(1, 'a' + i) + suffix;
            g->add_node(new HashableString(node_name));
            double curr_hash = HashableString(node_name).get_hash();
            g->move_node(curr_hash, vec4(i-4, 4, 0, 0));
            if(i > 0) {
                string prev_node_name = string(1, 'a' + i - 1) + suffix;
                double prev_hash = HashableString(prev_node_name).get_hash();
                g->add_edge(prev_hash, curr_hash);
                gs->config->set_edge_color(prev_hash, curr_hash, 0x00000000);
                gs->config->transition_edge_color(MICRO, prev_hash, curr_hash, 0xffffffff);
            }
        }
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("Here's two bad rankings for this graph."), 2);
    gs->manager.transition(MACRO, {
        {"y", to_string(left_graph_delta.y)},
        {"d", "20"},
    });
    for (char suffix : {'l', 'r'}) {
        unordered_map<string, int>& node_ranks = suffix == 'l' ? left_node_ranks : right_node_ranks;
        vec2 delta = suffix == 'l' ? left_graph_delta : right_graph_delta;
        for(int i = 0; i < 9; i++) {
            string node_name = string(1, 'a' + i) + suffix;
            double curr_hash = HashableString(node_name).get_hash();
            gs->transition_node_position(MICRO, curr_hash, vec4(i-4 + delta.x, node_ranks[node_name]-5 + delta.y, 0, 0));
        }
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("we don’t need to add any shortcuts if we just contract the nodes from left to right."), 17);
    gs->manager.transition(MACRO, {
        {"x", to_string(left_graph_delta.x)},
        {"y", to_string(left_graph_delta.y)},
        {"d", "14"},
    });
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        gs->config->transition_node_label(MICRO, HashableString(node_name).get_hash(), to_string(left_node_ranks[node_name]));
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("But when we search between the two ends,"), 3);
    // Splash a and h
    gs->render_microblock();
    gs->config->splash_node(HashableString("a").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("i").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("we have to explore every node and edge."), 9);
    trace_path(gs, {"a", "b", "c", "d", "e", "f", "g", "h", "i"}, 0xffff0000);

    stage_macroblock(FileBlock("No improvement from Dijkstra’s."), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Here’s another possible hierarchy."), 1);
    gs->manager.transition(MACRO, {
        {"x", to_string(right_graph_delta.x)},
        {"y", to_string(right_graph_delta.y)},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("with shortcuts over its lower triangles."), 6);
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        if(node_name == "d" || node_name == "e" || node_name == "c") continue;
        // Add shortcut to d
        g->add_edge(HashableString(node_name).get_hash(), HashableString("d").get_hash());
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("End to end, we only need to search two edges."), 10);
    gs->render_microblock();
    gs->config->splash_node(HashableString("a").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("i").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a", "d"}, 0xff00ff00);
    trace_path(gs, {"i", "d"}, 0xff00ff00);

    stage_macroblock(FileBlock("But we need a lot of shortcuts,"), 6);
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        if(node_name == "d" || node_name == "e" || node_name == "c") continue;
        gs->config->transition_edge_color(MICRO, HashableString(node_name).get_hash(), HashableString("d").get_hash(), 0xfffeffff);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("in some sections we still have to check all the edges."), 9);
    gs->render_microblock();
    gs->config->splash_node(HashableString("e").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("i").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"e", "f", "g", "h", "i"}, 0xffff0000);

    stage_macroblock(SilenceBlock(3), 1);
}

void render_video() {
    step1();
    //bad();
}
