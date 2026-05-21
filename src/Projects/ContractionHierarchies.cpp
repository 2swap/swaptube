#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

unordered_map<string, vec2> graph_nodes = {
    {"a", vec2(-1.5, .8)},
    {"b", vec2(0, 1)},
    {"c", vec2(-1.7, -.1)},
    {"d", vec2(-.8, .3)},
    {"e", vec2(.3, .2)},
    {"f", vec2(1.6, .3)},
    {"g", vec2(-.7, -.7)},
    {"h", vec2(.3, -.8)},
    {"i", vec2(1.6, -.8)},
};

unordered_map<string, int> graph_edges_with_weights = {
    {"ab", 1},
    {"ac", 1},
    {"be", 1},
    {"cd", 2},
    {"de", 3},

    {"ef", 1},
    {"eh", 3},
    {"fh", 1},
    {"hi", 7},

    {"gc", 10},
    {"ge", 5},
    {"gh", 4},
};

// First int is which step the node is contracted on, with 1 being the first contraction set, second being 2, etc. Nodes that aren't contracted have a negative value corresponding to which side of the contraction 1 split they lie on (2 is left, 3 is right)
// Second int is the rank
unordered_map<string, pair<int, int>> contraction_1 = {
    {"a", {2, 3}},
    {"b", {-2, 1}},
    {"c", {2, 4}},
    {"d", {-2, 2}},
    {"e", {1, 8}},
    {"f", {-3, 6}},
    {"g", {1, 9}},
    {"h", {3, 7}},
    {"i", {-3, 5}},
};

unordered_map<string, pair<int, int>> contraction_2 = {
    {"a", {2, 4}},
    {"b", {-2, 1}},
    {"c", {2, 3}},
    {"d", {-2, 2}},
    {"e", {1, 9}},
    {"f", {-3, 5}},
    {"g", {1, 8}},
    {"h", {3, 7}},
    {"i", {-3, 6}},
};

unordered_map<string, pair<int, int>> contraction_3 = {
    {"a", {-2, 1}},
    {"b", {1, 7}},
    {"c", {2, 3}},
    {"d", {-2, 2}},
    {"e", {1, 8}},
    {"f", {-3, 4}},
    {"g", {1, 9}},
    {"h", {3, 6}},
    {"i", {-3, 5}},
};

void split_graph_by_contraction(shared_ptr<GraphScene>& gs, unordered_map<string, pair<int, int>>& contraction, int step) {
    if(step == 0) {
        // Transition the nodes in contraction 1 to green
        for(auto& [node, data] : contraction) {
            if(data.first == 1) {
                gs->config->transition_node_color(MICRO, HashableString(node).get_hash(), 0xff00ff00);
            }
        }
        // Fade all edges between two nodes in contraction 1 to green
        for(auto& [edge, weight] : graph_edges_with_weights) {
            string node1 = edge.substr(0, 1);
            string node2 = edge.substr(1, 1);
            if(contraction[node1].first == 1 && contraction[node2].first == 1) {
                gs->config->fade_edge_color(MICRO, HashableString(node1).get_hash(), HashableString(node2).get_hash(), 0xff00ff00);
            }
        }
        gs->render_microblock();
    }

    if(step == 1) {
        // Transition edges which connect a node in contraction 1 to a node not in contraction 1 to very transparent white
        for(auto& [edge, weight] : graph_edges_with_weights) {
            string node1 = edge.substr(0, 1);
            string node2 = edge.substr(1, 1);
            if (contraction[node1].first == 1 && contraction[node2].first != 1) {
                gs->config->transition_edge_color(MICRO, HashableString(node1).get_hash(), HashableString(node2).get_hash(), 0x20ffffff);
            }
            else if (contraction[node1].first != 1 && contraction[node2].first == 1) {
                gs->config->transition_edge_color(MICRO, HashableString(node2).get_hash(), HashableString(node1).get_hash(), 0x20ffffff);
            }
        }
        gs->render_microblock();

        // Fade all left nodes (2 or -2) to red and all right nodes (3 or -3) to blue
        for(auto& [node, data] : contraction) {
            if(data.first == 2 || data.first == -2) {
                gs->config->transition_node_color(MICRO, HashableString(node).get_hash(), 0xffff0000);
            }
            else if(data.first == 3 || data.first == -3) {
                gs->config->transition_node_color(MICRO, HashableString(node).get_hash(), 0xff2020ff);
            }
        }
        // set all edges internal to the left side to red and all edges internal to the right side to blue
        for(auto& [edge, weight] : graph_edges_with_weights) {
            string node1 = edge.substr(0, 1);
            string node2 = edge.substr(1, 1);
            if(abs(contraction[node1].first) == 2 && abs(contraction[node2].first) == 2) {
                gs->config->set_edge_color(HashableString(node1).get_hash(), HashableString(node2).get_hash(), 0xffff0000);
            }
            else if(abs(contraction[node1].first) == 3 && abs(contraction[node2].first) == 3) {
                gs->config->set_edge_color(HashableString(node1).get_hash(), HashableString(node2).get_hash(), 0xff2020ff);
            }
        }
        gs->render_microblock();
    }

    if(step == 2) {
        // Label all nodes in contraction 1 with their rank
        for (auto& [node, data] : contraction) {
            if(data.first == 1) {
                gs->config->transition_node_label(MICRO, HashableString(node).get_hash(), to_string(data.second));
                gs->render_microblock();
            }
        }
    }

    if(step == 3) {
        for(auto& [edge, weight] : graph_edges_with_weights) {
            string node1 = edge.substr(0, 1);
            string node2 = edge.substr(1, 1);
            if (contraction[node1].first == 3 && contraction[node2].first == -3) {
                gs->config->transition_edge_color(MICRO, HashableString(node1).get_hash(), HashableString(node2).get_hash(), 0x20ffffff);
            }
            else if (contraction[node1].first == -3 && contraction[node2].first == 3) {
                gs->config->transition_edge_color(MICRO, HashableString(node2).get_hash(), HashableString(node1).get_hash(), 0x20ffffff);
            }
        }
        gs->render_microblock();

        // Transition the nodes in contraction -3 to light-blue
        for(auto& [node, data] : contraction) {
            if(data.first == -3) {
                gs->config->transition_node_color(MICRO, HashableString(node).get_hash(), 0xff8888ff);
            }
        }
        gs->render_microblock();
    }

    if(step == 4) {
        // Label the +3 nodes
        for (auto& [node, data] : contraction) {
            if(data.first == 3) {
                gs->config->transition_node_label(MICRO, HashableString(node).get_hash(), to_string(data.second));
                gs->render_microblock();
            }
        }
    }

    if(step == 5) {
        // Label the -3 nodes
        for (auto& [node, data] : contraction) {
            if(data.first == -3) {
                gs->config->transition_node_label(MICRO, HashableString(node).get_hash(), to_string(data.second));
                gs->render_microblock();
            }
        }
    }

    if(step == 6) {
        for(auto& [edge, weight] : graph_edges_with_weights) {
            string node1 = edge.substr(0, 1);
            string node2 = edge.substr(1, 1);
            if (contraction[node1].first == 2 && contraction[node2].first == -2) {
                gs->config->transition_edge_color(MICRO, HashableString(node1).get_hash(), HashableString(node2).get_hash(), 0x20ffffff);
            }
            else if (contraction[node1].first == -2 && contraction[node2].first == 2) {
                gs->config->transition_edge_color(MICRO, HashableString(node2).get_hash(), HashableString(node1).get_hash(), 0x20ffffff);
            }
        }
        gs->render_microblock();
        // Transition the nodes in contraction -2 to light-red
        for(auto& [node, data] : contraction) {
            if(data.first == -2) {
                gs->config->transition_node_color(MICRO, HashableString(node).get_hash(), 0xffff8888);
            }
        }
        gs->render_microblock();
    }

    if(step == 7) {
        // Label the +2 nodes
        for (auto& [node, data] : contraction) {
            if(data.first == 2) {
                gs->config->transition_node_label(MICRO, HashableString(node).get_hash(), to_string(data.second));
                gs->render_microblock();
            }
        }
    }

    if(step == 8) {
        // Label the -2 nodes
        for (auto& [node, data] : contraction) {
            if(data.first == -2) {
                gs->config->transition_node_label(MICRO, HashableString(node).get_hash(), to_string(data.second));
                gs->render_microblock();
            }
        }
    }
}

void add_shortcut(shared_ptr<Graph>& g, shared_ptr<GraphScene>& gs, string node1, string node2, string label) {
    double hash1 = HashableString(node1).get_hash();
    double hash2 = HashableString(node2).get_hash();
    g->add_edge(hash1, hash2);
    gs->config->set_edge_dashed(hash1, hash2, true);
    gs->config->set_edge_color(hash1, hash2, 0x000080ff);
    gs->config->transition_edge_color(MICRO, hash1, hash2, 0x400080ff);
    gs->config->transition_edge_label(MICRO, hash1, hash2, label);
}

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);

    gs->manager.set({
        {"q1", "1"},
        {"qi", "{t} .13 * sin .13 *"},
        {"qj", "{t} .11 * cos .13 *"},
        {"qk", "0"},
        {"d", "4"},
        {"edge_weights_size", "1.2"},
        {"points_radius_multiplier", "3"},
    });

    // Slide 47
    stage_macroblock(FileBlock("So let’s zoom in on a smaller graph to make sure that’s the case."), 1);
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

    stage_macroblock(FileBlock("These two nodes split the graph roughly in half,"), 3);
    split_graph_by_contraction(gs, contraction_1, 0);
    split_graph_by_contraction(gs, contraction_1, 1);

    stage_macroblock(FileBlock("so we give them the highest rank."), 2);
    split_graph_by_contraction(gs, contraction_1, 2);

    stage_macroblock(FileBlock("Looking at one side,"), 2);
    gs->render_microblock();
    gs->manager.transition(MICRO, {
        {"d", "3"},
        {"x", "1"},
        {"y", "-.5"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("this node splits the subset in half,"), 3);
    gs->config->splash_node(HashableString("h").get_hash());
    gs->render_microblock();
    split_graph_by_contraction(gs, contraction_1, 3);

    stage_macroblock(FileBlock("so it has the highest rank of the three."), 1);
    split_graph_by_contraction(gs, contraction_1, 4);

    stage_macroblock(FileBlock("We can rank the remaining two nodes in any order."), 2);
    split_graph_by_contraction(gs, contraction_1, 5);

    stage_macroblock(FileBlock("Now on the other side,"), 1);
    gs->manager.transition(MICRO, {
        {"d", "3"},
        {"x", "-1"},
        {"y", ".5"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("these two nodes split this subset in half,"), 3);
    gs->config->splash_node(HashableString("a").get_hash());
    gs->config->splash_node(HashableString("c").get_hash());
    gs->render_microblock();
    split_graph_by_contraction(gs, contraction_1, 6);

    stage_macroblock(FileBlock("so they have the next highest rank."), 2);
    split_graph_by_contraction(gs, contraction_1, 7);

    stage_macroblock(FileBlock("And then again, we rank the final two nodes with the remaining numbers."), 2);
    split_graph_by_contraction(gs, contraction_1, 8);

    stage_macroblock(SilenceBlock(1), 1);
    gs->manager.transition(MICRO, {
        {"d", "4"},
        {"x", "0"},
        {"y", "0"},
    });
    gs->render_microblock();

    // Slide 48
    stage_macroblock(FileBlock("But what about this ranking?"), 1);
    // Relabel all nodes with their rank in this new ranking
    for(auto& [node, data] : contraction_2) {
        gs->config->transition_node_label(MICRO, HashableString(node).get_hash(), to_string(data.second));
    }
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Or if we split the graph differently, we could get this ranking."), 16);
    for(int i = 0; i <= 8; i++)
        split_graph_by_contraction(gs, contraction_3, i);

    stage_macroblock(FileBlock("It even splits the graph perfectly in half."), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    // Splash all +2 or -2 nodes
    for(auto& [node, data] : contraction_3) {
        if(data.first == 2 || data.first == -2) {
            gs->config->splash_node(HashableString(node).get_hash());
        }
    }
    gs->render_microblock();
    // Splash all +3 or -3 nodes
    for(auto& [node, data] : contraction_3) {
        if(data.first == 3 || data.first == -3) {
            gs->config->splash_node(HashableString(node).get_hash());
        }
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("Isn’t that better?"), 1);
    gs->render_microblock();

    // Slide 49
    stage_macroblock(FileBlock("And these three rankings are all valid."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("We’ll see in a bit why it doesn’t matter which one we use,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("but for now we’ll take this one."), 16);
    for(int i = 0; i <= 8; i++)
        split_graph_by_contraction(gs, contraction_1, i);

    unordered_map<string, int> node_ranks = {};
    for(auto& [node, data] : contraction_1) {
        node_ranks[node] = data.second;
    }

    stage_macroblock(SilenceBlock(2), 3);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    for(auto& [node, coords] : graph_nodes) {
        string node_str = node;
        vec2 position = graph_nodes[node_str];
        gs->transition_node_position(MICRO, HashableString(node_str).get_hash(), vec4(position.x, position.y + 1.8, 0, 0));
    }
    gs->manager.transition(MICRO, "d", "7");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("The best way to visualize this ranking is to pull down or contract the nodes in order."), 1);
    gs->render_microblock();

    unordered_map<string, vec2> contracted_positions = {
        {"a", vec2(-.5, 3)},
        {"b", vec2(-.1, 1)},
        {"c", vec2(-1.5, 4)},
        {"d", vec2(-1.3, 2)},
        {"e", vec2(.4, 8)},
        {"f", vec2(.8, 6)},
        {"g", vec2(.5, 10)},
        {"h", vec2(1.4, 7.4)},
        {"i", vec2(1.5, 5)},
    };

    stage_macroblock(CompositeBlock(FileBlock("1 is the furthest down, 2 is next up and so on."), SilenceBlock(3)), 11);
    for (int rank = 1; rank <= 9; rank++) {
        // Find node with this rank
        string node;
        for(auto& [n, r] : node_ranks) {
            if(r == rank) {
                node = n;
                break;
            }
        }
        gs->transition_node_position(MICRO, HashableString(node).get_hash(), vec4(contracted_positions[node].x * 1.8, contracted_positions[node].y / 2.5 - 2, 0, 0));
        if(node == "b" || node == "d") {
            gs->render_microblock();
        }
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("We’ll leave the original graph at the top for reference."), 2);
    // Create a graph with the original placement but translated by (-2.8,2)
    // Create nodes with primes (') such as c', d', etc. for this original reference graph
    shared_ptr<Graph> mini_g = make_shared<Graph>();
    shared_ptr<GraphScene> mini_gs = make_shared<GraphScene>(mini_g, vec2(.5,.5));
    //CompositeScene cs;
    //cs.add_scene(gs, "gs");
    for (auto& [node, coords] : graph_nodes) {
        string node_str = node + "'";
        vec4 position = vec4(coords.x / 2 - 2.8, coords.y / 2 + 2, 0, 1);
        mini_g->add_node(new HashableString(node_str));
        mini_g->move_node(HashableString(node_str).get_hash(), position);
        mini_gs->config->add_node_if_missing(HashableString(node_str).get_hash());
        mini_gs->config->set_node_radius(HashableString(node_str).get_hash(), 0.5);
    }
    for (auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1) + "'";
        string node2 = edge.substr(1, 1) + "'";
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        mini_g->add_edge(hash1, hash2);
        mini_gs->config->set_edge_label(hash1, hash2, to_string(weight));
    }
    gs->render_microblock();
    //cs.add_scene_fade_in(MICRO, mini_gs, "mini_gs");
    gs->render_microblock();

    // Slide 50
    stage_macroblock(FileBlock("Now we can bring in a search algorithm."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("To speed up the runtime, we limit the search space by only moving up the hierarchy."), 1);
    // Transition all edge colors from side of low rank to side of high rank
    for (auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        string lower_rank_node = node_ranks[node1] < node_ranks[node2] ? node1 : node2;
        string higher_rank_node = node_ranks[node1] < node_ranks[node2] ? node2 : node1;
        double hash1 = HashableString(lower_rank_node).get_hash();
        double hash2 = HashableString(higher_rank_node).get_hash();
        gs->config->transition_edge_color(MICRO, hash1, hash2, 0xffff0000);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("And that’s why it’s so important to use a bidirectional Dijkstra."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("If the algorithm could only search in one direction,"), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"c", "b", "h"}, 0xfffffffe);

    stage_macroblock(FileBlock("it would get stuck on a lot of pairs- it can only go up and never back down."), 6);
    // Highlight pair with ranks 4 and 7 (c and h)
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xff8080ff);
    gs->config->transition_node_color(MICRO, HashableString("h").get_hash(), 0xff8080ff);
    trace_path(gs, {"c", "b"}, 0xff00ff00);
    trace_path(gs, {"b", "h"}, 0xffff0000);
    gs->render_microblock();

    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    // Slide 51
    stage_macroblock(FileBlock("Let’s add the edge weights back."), 1);
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(weight));
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("What’s the shortest path from node 4 to node 8?"), 5);
    // Highlight C and E (ranks 4 and 8)
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xffff0000);
    gs->config->transition_node_color(MICRO, HashableString("c'").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("e").get_hash(), 0xff00ff00);
    gs->config->transition_node_color(MICRO, HashableString("e'").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Well, starting from node 4, we search up and reach node 9 with a cost of 10."), 4);
    gs->render_microblock();
    trace_path(gs, {"c", "g"}, 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("From node 8, we also search up and get to node 9 with a cost of 5."), 2);
    // Add edge e to g
    gs->config->transition_edge_color(MICRO, HashableString("e").get_hash(), HashableString("g").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->splash_node(HashableString("g").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("So, 4-3-1-8 must be the shortest path with a cost of 15. Done."), 15);
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"c", "g", "e"}, 0xffff0000);
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

    stage_macroblock(FileBlock("Except there's a problem, because that’s not the shortest path."), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Using this path only costs 3."), 16);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"c", "a", "b", "e"}, 0xff00ff00);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    // Slide 52
    stage_macroblock(FileBlock("This search doesn’t check paths that go through lower ranked nodes."), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("But we can fix it with a little more preprocessing."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Start from the lowest rank. Does this node connect to at least two higher nodes?"), 4);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("b").get_hash(), 0xffff8080);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("On this graph, node 1 connects to 3 and 8."), 2);
    gs->config->transition_edge_color(MICRO, HashableString("b").get_hash(), HashableString("a").get_hash(), 0xffff8080);
    gs->config->transition_edge_color(MICRO, HashableString("b").get_hash(), HashableString("e").get_hash(), 0xffff8080);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xffff8080);
    gs->config->transition_node_color(MICRO, HashableString("e").get_hash(), 0xffff8080);
    gs->render_microblock();

    stage_macroblock(FileBlock("This is called a lower triangle."), 3);
    trace_path(gs, {"a", "b", "e"}, 0xffff0000);

    stage_macroblock(FileBlock("And while there’s a chance this path is actually the shortest between node 3 and node 8,"), 9);
    // Splash A and E
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("a").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("e").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("our search will never consider it."), 2);
    // Pop B
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    // Slide 53
    stage_macroblock(FileBlock("So, we’ll add a shortcut between node 3 and node 8"), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    add_shortcut(g, gs, "a", "e", "");
    gs->render_microblock();

    stage_macroblock(FileBlock("that’s just the cost of this lower triangle."), 5);
    gs->render_microblock();
    gs->config->transition_edge_label(MICRO, HashableString("a").get_hash(), HashableString("e").get_hash(), "\\text{5 (via B)}");
    gs->render_microblock();
    trace_path(gs, {"a", "b", "e"}, 0xffff0000);

    stage_macroblock(FileBlock("This edge doesn’t correspond to anything real on the road network."), 5);
    gs->config->fade_edge_color(MICRO, HashableString("a").get_hash(), HashableString("e").get_hash(), 0x800080ff);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("It’s just a way to keep track of shortest paths."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("We move on to the next lowest node and run the same test."), 6);
    gs->render_microblock();
    // Splash D
    gs->config->splash_node(HashableString("d").get_hash());
    gs->render_microblock();
    // Trace paths up to higher nodes c and e
    trace_path(gs, {"d", "c"}, 0xffff8080);
    trace_path(gs, {"d", "e"}, 0xffff8080);

    stage_macroblock(FileBlock("But there could be multiple lower triangles,"), 8);
    trace_path(gs, {"c", "d", "e"}, 0xffff0000);
    // TODO do not clear the shortcut's cyan color
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"c", "a", "e"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("or there might already be an edge between the two higher nodes."), 7);
    trace_path(gs, {"e", "h"}, 0xff00ff00);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    trace_path(gs, {"e", "f", "h"}, 0xffff0000);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("In those cases, the shortcut represents whichever path has the minimum cost."), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    // Add shortcut from c to e with cost 5 via d
    add_shortcut(g, gs, "c", "e", "\\text{5 (via D)}");
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_edge_color(MICRO, HashableString("e").get_hash(), HashableString("h").get_hash(), 0x00ffffff);
    gs->render_microblock();
    add_shortcut(g, gs, "e", "h", "\\text{2 (via F)}");

    stage_macroblock(FileBlock("A path that only uses lower ranked nodes is like a path that only uses small, local roads."), 1);
    gs->render_microblock();
    return;

    stage_macroblock(FileBlock("We don’t want the algorithm to return a highway route if a local road is shorter."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Building shortcuts from the bottom up makes sure we don’t miss these paths while also ignoring them if they aren’t needed. This way, we both limit the search space and never miss the shortest path."), 1);
    // Fade out shortcut and then remove
    gs->config->fade_edge_color(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00000000);
    gs->config->transition_edge_label(MICRO, HashableString("a'").get_hash(), HashableString("c'").get_hash(), "");
    gs->render_microblock();
    g->remove_edge(HashableString("a'").get_hash(), HashableString("c'").get_hash());
    gs->config->set_edge_color(HashableString("a'").get_hash(), HashableString("c'").get_hash(), 0x00ffffff);

    // Slide 54
    stage_macroblock(FileBlock("There’s another benefit to shortcuts. We don’t need to worry too much about getting a perfect ranking."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("We can roughly cut the graph in half and rank nodes in the same cut in any order."), 11);
    gs->render_microblock();
    // Splash node b
    gs->config->splash_node(HashableString("b'").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"b'", "a'"}, 0xffff0001);
    trace_path(gs, {"b'", "c'"}, 0xffff0001);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("If there was a shortest path we missed, we’ll add back later with a shortcut."), 4);
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

    stage_macroblock(FileBlock("That’s why we were able to get three valid rankings using the same instructions earlier."), 15);
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
}
