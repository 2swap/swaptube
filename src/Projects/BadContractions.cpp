#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"q1", "1"},
        {"qi", "{t} .09 * sin .08 *"},
        {"qj", "{t} .07 * cos .08 *"},
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

    stage_macroblock(SilenceBlock(.5), 1);
    for(int i = 0; i < 9; i++) {
        string node_name = string(1, 'a' + i);
        g->add_node(new HashableString(node_name));
        double curr_hash = HashableString(node_name).get_hash();
        g->move_node(curr_hash, vec4(i-4, 4, 0, 0));
        gs->config->set_node_color(curr_hash, 0x00000000);
        if(i > 0) {
            string prev_node_name = string(1, 'a' + i - 1);
            double prev_hash = HashableString(prev_node_name).get_hash();
            g->add_edge(prev_hash, curr_hash);
            gs->config->set_edge_color(prev_hash, curr_hash, 0x00000000);
        }
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, with a series of nodes in a straight line,"), 17);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a", "b", "c", "d", "e", "f", "g", "h", "i"}, 0xffffffff);

    stage_macroblock(FileBlock("we don’t need to add any shortcuts if we just contract the nodes from left to right."), 29);
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
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        gs->config->transition_node_label(MICRO, HashableString(node_name).get_hash(), to_string(node_ranks[node_name]));
        gs->transition_node_position(MICRO, HashableString(node_name).get_hash(), vec4(i-4, node_ranks[node_name]-5, 0, 0));
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("But when we search between the two ends,"), 7);
    // Splash a and h
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->splash_node(HashableString("a").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("i").get_hash());
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(SilenceBlock(.5), 1);
    gs->manager.transition(MICRO, {
        {"d", "12"},
        {"x", "-2"},
        {"y", "-2"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("we have to explore every node and edge."), 9);
    gs->manager.transition(MACRO, {
        {"d", "13"},
        {"x", "2"},
        {"y", "2"},
    });
    trace_path(gs, {"a", "b", "c", "d", "e", "f", "g", "h", "i"}, 0xffff0000);

    stage_macroblock(FileBlock("No improvement over Dijkstra’s."), 1);
    gs->manager.transition(MACRO, {
        {"d", "14"},
        {"x", "0"},
        {"y", "0"},
    });
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

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

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Here’s another possible ranking."), 1);
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        gs->config->transition_node_label(MICRO, HashableString(node_name).get_hash(), to_string(node_ranks[node_name]));
        gs->transition_node_position(MICRO, HashableString(node_name).get_hash(), vec4(i-4, node_ranks[node_name]-5, 0, 0));
    }
    StateSet undo = gs->manager.transition(MACRO, {
        {"qj", "-.07"},
        {"x", "-1"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("with the necessary shortcuts."), 6);
    gs->manager.transition(MACRO, {
        {"qj", ".07"},
        {"x", "1"},
    });
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        if(node_name == "d" || node_name == "e" || node_name == "c") continue;
        // Add shortcut to d
        g->add_edge(HashableString(node_name).get_hash(), HashableString("d").get_hash());
        gs->config->add_edge_if_missing(HashableString(node_name).get_hash(), HashableString("d").get_hash());
        gs->config->set_edge_dashed(HashableString(node_name).get_hash(), HashableString("d").get_hash(), true);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("End to end, we only need to search two edges."), 12);
    gs->manager.transition(MACRO, undo);
    gs->render_microblock();
    gs->config->splash_node(HashableString("a").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("i").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"a", "d"}, 0xff00ff00);
    trace_path(gs, {"i", "d"}, 0xff00ff00);

    stage_macroblock(FileBlock("But we need a lot of shortcuts,"), 9);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    for(int i = 0; i < 9; i++) {
        string node_name(1, 'a' + i);
        if(node_name == "d" || node_name == "e" || node_name == "c") continue;
        gs->config->fade_edge_color(MICRO, HashableString(node_name).get_hash(), HashableString("d").get_hash(), 0xffff0000);
        gs->render_microblock();
    }
    gs->render_microblock();
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("and in some sections we still have to check all the edges."), 9);
    gs->manager.transition(MACRO, {
        {"y", "-2"},
        {"x", "2"},
    });
    gs->render_microblock();
    gs->config->splash_node(HashableString("e").get_hash());
    gs->render_microblock();
    gs->config->splash_node(HashableString("i").get_hash());
    gs->render_microblock();
    gs->render_microblock();
    trace_path(gs, {"e", "f", "g", "h", "i"}, 0xffff0000);

    stage_macroblock(SilenceBlock(2), 2);
    gs->render_microblock();
    gs->render_microblock();
}
