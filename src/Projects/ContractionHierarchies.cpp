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
    {"cd", 2},
    {"ae", 10},
    {"ef", 1},
    {"fh", 7},
    {"ec", 5},
    {"cf", 1},
    {"fd", 8},
};

void step1() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);

    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "4"},
        {"edge_weights_size", "1"},
        {"points_radius_multiplier", "3"},
    });

    stage_macroblock(FileBlock("Let's zoom in on a smaller part of the graph to see how this really works."), 1);
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
    stage_macroblock(FileBlock("We can ignore the edge weights for the first step."), 1);
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
    gs->render_microblock();
    // Make left side green
    gs->config->set_node_color(HashableString("a").get_hash(), 0xff00ff00);
    gs->config->set_node_color(HashableString("b").get_hash(), 0xff00ff00);
    gs->config->set_edge_color(HashableString("a").get_hash(), HashableString("b").get_hash(), 0xff00ff00);
    gs->render_microblock();
    // Make right side blue
    gs->config->set_node_color(HashableString("d").get_hash(), 0xff0088ff);
    gs->config->set_node_color(HashableString("f").get_hash(), 0xff0088ff);
    gs->config->set_node_color(HashableString("h").get_hash(), 0xff0088ff);
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

    stage_macroblock(FileBlock("so they will have the highest rank."), 2);
    gs->render_microblock();
    gs->config->transition_node_label(MICRO, HashableString("c").get_hash(), to_string(node_ranks["c"]));
    gs->config->transition_node_label(MICRO, HashableString("e").get_hash(), to_string(node_ranks["e"]));
    gs->render_microblock();

    // Fade all nodes and edges gray except for d, f and h
    stage_macroblock(FileBlock("We can split one side a little further, so that node gets the next highest rank."), 4);
    for(auto& [node, coords] : graph_nodes) {
        if(node == "d" || node == "f" || node == "h") continue;
        gs->config->fade_node_color(MICRO, HashableString(node).get_hash(), 0xff505050);
    }
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        if((node1 == "d" || node1 == "f" || node1 == "h") && (node2 == "d" || node2 == "f" || node2 == "h")) continue;
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        gs->config->fade_edge_color(MICRO, hash1, hash2, 0xff505050);
    }
    gs->render_microblock();

    // Highlight f
    gs->config->transition_node_color(MICRO, HashableString("f").get_hash(), 0xff00ff00);
    gs->config->transition_node_label(MICRO, HashableString("f").get_hash(), to_string(node_ranks["f"]));
    gs->render_microblock();

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

    stage_macroblock(FileBlock("It’s easier to see this hierarchy in 3D."), 3);
    // Duplicate the whole graph
    for(auto& [node, coords] : graph_nodes) {
        double hash = HashableString(node).get_hash();
        vec4 position = vec4(coords.x, coords.y, 0, 1);
        g->add_node(new HashableString(node + "'"));
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
        gs->transition_node_position(MICRO, HashableString(node_str).get_hash(), vec4(position.x, position.y + 1.5, 0, 0));
        gs->transition_node_position(MICRO, HashableString(node_str + "'").get_hash(), vec4(position.x, position.y - .8, 0, 0));
    }
    gs->manager.transition(MICRO, "d", "7");
    gs->render_microblock();

    for(auto& [node, coords] : graph_nodes) {
        string node_str = node;
        vec2 position = graph_nodes[node_str];
        gs->transition_node_position(MICRO, HashableString(node_str + "'").get_hash(), vec4(position.x, .5, position.y, 0));
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("The original graph is in white, and then the nodes are pulled down, or contracted, by their rank."), 1);
    gs->render_microblock();

    unordered_map<string, vec2> contracted_positions = {
        {"a", vec2(-1.5, 2)},
        {"b", vec2(.7, 1)},
        {"c", vec2(.2, 6)},
        {"d", vec2(1, 4)},
        {"e", vec2(-.2, 7.5)},
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

    stage_macroblock(FileBlock("Eventually, our algorithm will use a bidirectional search,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("but it will only ever search from low ranks to higher ranks."), 1);
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

    stage_macroblock(FileBlock("But if we search like that, this graph has an obvious problem."), 1);
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

    stage_macroblock(FileBlock("What’s the shortest path from A to C using this hierarchy search?"), 5);
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

    stage_macroblock(FileBlock("Shortest path is A, E, C with a cost of 15. Done."), 7);
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "e'", "c'"}, 0xffff0001);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Except that’s not the shortest path."), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Using the path A, B, C only costs 5."), 7);
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a", "b", "c"}, 0xff00ff00);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("What went wrong?"), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("B has edges to two higher ranked nodes."), 3);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("b'").get_hash(), 0xffff0000);
    gs->config->transition_edge_color(MICRO, HashableString("b'").get_hash(), HashableString("a'").get_hash(), 0xffff0000);
    gs->config->transition_edge_color(MICRO, HashableString("b'").get_hash(), HashableString("c'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("But since it’s the lowest of the three, there’s no way to consider that path."), 3);
    // Pop B
    gs->render_microblock();
    gs->config->splash_node(HashableString("b'").get_hash());
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Remember, our search never follows a path that’s up, down, up."), 3);
    // Highlight path A, B, C
    trace_path(gs, {"a'", "b'", "c'"}, 0xff00ff00);

    stage_macroblock(FileBlock("To fix it, we’ll add a shortcut between A and C."), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    g->add_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x2000ffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Its weight is just the cost of the path A, B, C and it has a midpoint marker to B."), 5);
    trace_path(gs, {"a'", "b'", "c'"}, 0xff00ff00);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "5 (b)");
    gs->render_microblock();

    stage_macroblock(FileBlock("The bidirectional search now checks both the upper path [AEC] and the shortcut to find the best path."), 7);
    trace_path(gs, {"a'", "e'", "c'"}, 0xff00ff00);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"a'", "c'"}, 0xff00ff00);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("But there's three more problems if we want this to work on ANY graph."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("First, on a slightly larger graph,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("it’s possible this node wasn’t the only lower ranked node connected up to these two nodes."), 7);
    gs->render_microblock();
    // Splash node b
    gs->config->splash_node(HashableString("b'").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("a'").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("c'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("There could be one, or two, or even more."), 5);
    // Add nodes i and j, each connected to c and a
    gs->render_microblock();
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
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    // transition path a' i' c'
    gs->render_microblock();
    trace_path(gs, {"a'", "i'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    // transition path a' j' c'
    gs->render_microblock();
    trace_path(gs, {"a'", "j'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("So in general, the shortcut is the minimum cost of all lower triangles."), 2);
    gs->render_microblock();
    g->remove_node(HashableString("i'").get_hash());
    g->remove_node(HashableString("j'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Second, it’s possible the shortest path between these two nodes goes through multiple lower ranked nodes."), 5);
    gs->render_microblock();
    // Add node k and l
    g->add_node(new HashableString("k'"));
    g->move_node(HashableString("k'").get_hash(), vec4(contracted_positions["k"].x, contracted_positions["k"].y / 3. - 2, 0, 0));
    gs->render_microblock();
    g->add_node(new HashableString("l'"));
    g->move_node(HashableString("l'").get_hash(), vec4(contracted_positions["l"].x, contracted_positions["l"].y / 3. - 2, 0, 0));
    // Connect a-k-l-c
    g->add_edge(HashableString("a'").get_hash(), HashableString("k'").get_hash());
    gs->render_microblock();
    g->add_edge(HashableString("k'").get_hash(), HashableString("l'").get_hash());
    gs->render_microblock();
    g->add_edge(HashableString("l'").get_hash(), HashableString("c'").get_hash());
    gs->config->set_edge_color(HashableString("l'").get_hash(), HashableString("c'").get_hash(), 0x0000000);
    gs->config->transition_edge_color(MICRO, HashableString("l'").get_hash(), HashableString("c'").get_hash(), 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, this is the shortest path, but it passes through two lower ranked nodes, so it doesn’t form a lower triangle."), 10);
    trace_path(gs, {"a'", "k'", "l'", "c'"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("k'").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("l'").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Here's what we'll do:"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("In this example, the lowest node does connect up to two higher nodes,"), 3);
    gs->render_microblock();
    gs->config->splash_node(HashableString("k'").get_hash());
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, HashableString("k'").get_hash(), HashableString("a'").get_hash(), 0xfffffffe);
    gs->config->transition_edge_color(MICRO, HashableString("k'").get_hash(), HashableString("l'").get_hash(), 0xfffffffe);
    gs->render_microblock();

    stage_macroblock(FileBlock("so we add a shortcut between them."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Like before, we find minimum cost over the lower triangles to get the shortcut weight."), 3);
    trace_path(gs, {"a'", "k'", "l'"}, 0xffff0000);

    stage_macroblock(FileBlock("There's only one triangle, so it's just 2."), 2);
    g->add_edge(HashableString("a'").get_hash(), HashableString("l'").get_hash());
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("l'").get_hash(), 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("l'").get_hash(), 0x2000ffff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("l'").get_hash(), "2()");
    gs->render_microblock();

    stage_macroblock(FileBlock("Now node 2 forms a lower triangle!"), 3);
    trace_path(gs, {"a'", "l'", "c'"}, 0xffff0000);

    stage_macroblock(FileBlock("We add a shortcut and compare the two lower triangles to get the minimum cost."), 10);
    g->add_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->transition_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x2000ffff);
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a'", "k'", "c'"}, 0xffff0000);
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "5()");
    gs->render_microblock();

    stage_macroblock(FileBlock("Finally, the shortcut between these two nodes points to the right path."), 6);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Adding shortcuts from the bottom up makes sure we don’t miss local routes that are better than the highway."), 19);
    trace_path(gs, {"a'", "k'", "l'"}, 0xffff0000);
    trace_path(gs, {"a'", "l'"}, 0xff0080ff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"a'", "l'", "c'"}, 0xffff0000);
    trace_path(gs, {"a'", "c'"}, 0xff0080ff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("After all, the shortest path will have the minimum cost,"), 3);
    trace_path(gs, {"a'", "b'", "c'"}, 0xffff0000);

    stage_macroblock(FileBlock("and the minimum cost ‘bubbles up’ as we build shortcuts."), 14);
    trace_path(gs, {"a'", "k'", "l'"}, 0xffff0000);
    trace_path(gs, {"a'", "l'"}, 0xff0080ff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"a'", "l'", "c'"}, 0xffff0000);
    trace_path(gs, {"a'", "c'"}, 0xff0080ff);
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("If a lower path isn’t the minimum, then we discard early and we don’t need to check it again."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Onto the third problem."), 1);
    g->remove_node(HashableString("k'").get_hash());
    g->remove_node(HashableString("l'").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Remember that we found the node ranking without considering edge weights. They could’ve been anything."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("See if you spot the problem when we change a few weights."), 7);
    // c-d to 1
    // c-f to 3
    // d-f to 1
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("c").get_hash(), HashableString("d").get_hash(), "1");
    gs->config->transition_edge_label(MICRO, HashableString("c'").get_hash(), HashableString("d'").get_hash(), "1");
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("c").get_hash(), HashableString("f").get_hash(), "3");
    gs->config->transition_edge_label(MICRO, HashableString("c'").get_hash(), HashableString("f'").get_hash(), "3");
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("d").get_hash(), HashableString("f").get_hash(), "1");
    gs->config->transition_edge_label(MICRO, HashableString("d'").get_hash(), HashableString("f'").get_hash(), "1");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

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

    stage_macroblock(FileBlock("But we need to run the same bottom up check for any lower triangle, even if the edge already exists in the graph."), 4);
    // f d c is a lower triangle, even though edge c-f exists
    trace_path(gs, {"f'", "d'", "c'"}, 0xffff0000);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("f'").get_hash(), HashableString("c'").get_hash(), "2");

    stage_macroblock(FileBlock("In this case, we replace these two edges with shortcuts that are the minimum cost over the lower triangles."), 4);
    // c' f' e'
    trace_path(gs, {"c'", "f'", "e'"}, 0xffff0000);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("c'").get_hash(), HashableString("e'").get_hash(), "3");

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

    unordered_map<string, int> node_ranks = {
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
    stage_macroblock(FileBlock("This graph has a bad ranking."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("For a linear sequence of nodes,"), 9);

    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        g->add_node(new HashableString(node_name));
        g->move_node(HashableString(node_name).get_hash(), vec4(i-4, 0, 0, 0));
        if(i > 0) {
            g->add_edge(HashableString(string(1, 'a' + i - 1)).get_hash(), HashableString(node_name).get_hash());
        }
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("we don’t need to add any shortcuts if we just contract the nodes from left to right."), 17);
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
        gs->transition_node_position(MICRO, HashableString(node_name).get_hash(), vec4(i-4, node_ranks[node_name]-5, 0, 0));
        gs->config->transition_node_label(MICRO, HashableString(node_name).get_hash(), to_string(node_ranks[node_name]));
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
    gs->render_microblock();

    stage_macroblock(FileBlock("Here’s another possible ranking."), 9);
    node_ranks = {
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

    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        gs->transition_node_position(MICRO, HashableString(node_name).get_hash(), vec4(i-4, node_ranks[node_name]-5, 0, 0));
        gs->config->transition_node_label(MICRO, HashableString(node_name).get_hash(), to_string(node_ranks[node_name]));
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("It needs a lot of shortcuts,"), 8);
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        if(node_name == "d") continue;
        // Add shortcut to d
        g->add_edge(HashableString(node_name).get_hash(), HashableString("d").get_hash());
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("but end to end, we only need to search two edges."), 3);

}

void render_video() {
    step1();
    bad();
}
