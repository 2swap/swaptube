#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"

// Lists of KlotskiBoards
auto gpt = {gpt2, gpt3, weird1, weird2, weird3};
auto other = {apk, mathgames_12_13_04, euler766_easy};
auto suns = {fatsun, sun_no_minis, truncatedsun};
auto rushhours = {beginner, intermediate, advanced, expert, reddit, guh3, guh4, video, thinkfun1, thinkfun2, thinkfun3};
auto big = {sun};
auto geometry = {jam3x3, cube_4d, cube_6d, big_block, diamond, doublering, outer_ring, plus_3_corners, plus_4_corners, ring, ring_big, rows, small_block, t_shapes, triangles, triangles2};

// TODO animate this function by implementing the comments. Do not delete the comments.
void step1(){
    CompositeScene cs;

    // Add KlotskiScene and GraphScene for Intermediate puzzle, begin making 50 random moves.
    KlotskiScene ks(intermediate);
    StateSet board_width_height{{"w",".3"},{"h",to_string(.3*VIDEO_WIDTH/VIDEO_HEIGHT)}};
    StateSet board_position    {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}};
    ks.state_manager.set(board_width_height);
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks.copy_board()));
    GraphScene gs(&g);
    gs.state_manager.set({
        {"q1", "1"},
        {"qi", "<t> .1 * cos"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"}, // Camera orientation quaternion
        {"decay",".8"},
        {"surfaces_opacity","0"}, // Whether we want to draw the board at every node
        {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
    });
    cs.add_scene(&gs, "gs");
    cs.add_scene(&ks, "ks");
    cs.state_manager.set(board_position);
    cs.inject_audio(FileSegment("What you're looking at is a random agent exploring the state-space graph of a slidy puzzle."), 200);
    while(cs.microblocks_remaining()) {
        ks.stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g.add_missing_edges(true);
        // Highlight the node of the board on the state-space graph
        gs.next_hash = ks.copy_board().get_hash();
        cs.render(); // Render a microblock
    }

    // Set ks.highlight_char to highlight the block which needs to get freed.
    cs.inject_audio(FileSegment("It wants to free this block from the hole on the right side of the board."), 4);
    cs.state_manager.microblock_transition({{"ks.x",".5"},{"ks.y",".5"}});
    ks.state_manager.microblock_transition({{"w","1"},{"h","1"}});
    cs.render();
    ks.highlight_char = 'b';
    cs.render();
    ks.stage_move('b', 4, 0);
    cs.render();
    ks.stage_move('b', -4, 0);
    cs.render();

    ks.highlight_char = 'a';
    cs.inject_audio_and_render(FileSegment("However, this piece is in the way..."));

    cs.state_manager.microblock_transition(board_position);
    ks.state_manager.microblock_transition(board_width_height);
    cs.inject_audio_and_render(SilenceSegment(1));

    // Delete all nodes of the graph except for the current one. Turn on surface opacity, and turn off edge opacity.
    cs.inject_audio(SilenceSegment(3), g.size()-1);
    double kb_to_keep = ks.copy_board().get_hash();
    gs.state_manager.macroblock_transition({{"surfaces_opacity",".5"},{"lines_opacity","0"}});
    gs.state_manager.set({{"centering_strength","0"}});
    unordered_map<double,Node> nodes_copy = g.nodes;
    for(auto it = nodes_copy.begin(); it != nodes_copy.end(); ++it){
        double id_here = it->first;
        if(id_here == kb_to_keep) continue;
        g.remove_node(id_here);
        cs.render();
    }
    g.clear_queue();

    // Re-center
    gs.state_manager.microblock_transition({{"centering_strength","1"}});
    cs.inject_audio_and_render(FileSegment("We'll represent the current position of the puzzle as a single node."));

    // Make one move and insert it on the graph.
    ks.stage_random_move();
    ks.highlight_char = 'c';
    cs.inject_audio_and_render(FileSegment("If we make a single move on the puzzle,"));
    g.add_to_stack(new KlotskiBoard(ks.copy_staged_board()));
    g.add_missing_edges(true);
    gs.next_hash = ks.copy_board().get_hash();
    cs.inject_audio_and_render(FileSegment("we arrive at a different node."));

    // No-op
    cs.inject_audio_and_render(FileSegment("Since these two positions are only separated by a single move,"));

    // Turn edge opacity on.
    gs.state_manager.macroblock_transition({{"lines_opacity","1"}});
    cs.inject_audio_and_render(FileSegment("We draw an edge connecting their two nodes."));

    // Make a few random moves.
    gs.state_manager.macroblock_transition({{"surfaces_opacity","0"}});
    cs.inject_audio(FileSegment("Each node is connected to a few more, and drawing them, we construct this labyrinth of paths."), 8);
    while(cs.microblocks_remaining()) {
        ks.stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g.add_missing_edges(true);
        gs.next_hash = ks.copy_board().get_hash();
        cs.render();
    }

    // Unhighlight and fade out the KlotskiScene
    gs.next_hash = 0;
    cs.state_manager.macroblock_transition({{"ks.opacity","0"}});
    cs.inject_audio_and_render(SilenceSegment(2));

    // Expand the graph by one node until it is complete.
    int g_size;
    {
        Graph g2;
        g2.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g2.expand_completely();
        g_size = g2.size();
    }
    cs.inject_audio(FileSegment("If we grow out the graph in its entirety..."), g_size);
    while(cs.microblocks_remaining()) {
        g.expand_once();
        cs.render();
    }

    //TODO highlight the solutions
    cs.inject_audio(FileSegment("we can see that the random agent is still quite far away from the solutions."), g_size);

    // No-op, the topology should be apparent.
    cs.inject_audio_and_render(FileSegment("We can also make a lot more observations about the topology of this puzzle!"));

    // Fade out everything from the CompositeScene and then delete.
    cs.fade_out_all_scenes();
    cs.inject_audio_and_render(FileSegment("But, let's start simple."));
    cs.remove_all_scenes();
}

void step2() {
    CompositeScene cs;
    // Start over by adding a KlotskiScene and GraphScene.
    KlotskiScene ks2(beginner, .5, .5*VIDEO_WIDTH/VIDEO_HEIGHT);
    cs.add_scene_fade_in(&ks2, "ks2", 0.25, 0.25*VIDEO_WIDTH/VIDEO_HEIGHT);
    Graph g2;
    g2.add_to_stack(new KlotskiBoard(ks2.copy_board()));
    GraphScene gs2(&g2);
    cs.add_scene_fade_in(&gs2, "gs2");
    cs.inject_audio_and_render(FileSegment("This is a beginner's puzzle."));
}

void render_video() {
    step1();
    step2();
}
