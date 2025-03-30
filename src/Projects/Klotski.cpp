#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"

void demo(){
    //FOR_REAL = false;
    //PRINT_TO_TERMINAL = false;

    // Lists of KlotskiBoards
    auto gpt = {gpt2, gpt3, weird1, weird2, weird3};
    auto other = {apk, mathgames_12_13_04, euler766_easy};
    auto suns = {fatsun, sun_no_minis, truncatedsun};
    auto rushhours = {beginner, intermediate, advanced, expert, reddit, guh3, guh4, video, thinkfun1, thinkfun2, thinkfun3};
    auto big = {sun};
    auto geometry = {jam3x3, cube_4d, cube_6d, big_block, diamond, doublering, outer_ring, plus_3_corners, plus_4_corners, ring, ring_big, rows, small_block, t_shapes, triangles, triangles2};

    for(KlotskiBoard kb : {apk}){
        KlotskiScene ks(kb.w, kb.h, kb.representation, kb.rushhour, .5, .5*VIDEO_WIDTH/VIDEO_HEIGHT);
        CompositeScene cs;
        cs.add_scene(&ks, "ks", 0.25, 0.25*VIDEO_WIDTH/VIDEO_HEIGHT);

        Graph g;
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        Graph g2;
        g2.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g2.expand_completely();
        GraphScene gs(&g);
        gs.state_manager.set({
            {"q1", "0"},
            {"qi", "<t> .1 * cos"},
            {"qj", "<t> .1 * sin"},
            {"qk", "0"},
            {"d", "1"},
            {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
            {"attract","1"},
            {"repel","1"},
            {"decay",".8"},
            {"surfaces_opacity","0"}, // 0 meaning do not render the corresponding board on top of each node.
            {"points_opacity","1"},
            {"lines_opacity","1"},
        });
        cs.add_scene(&gs, "gs");

        // Let a random agent make move around on the board and build the graph as it does
        cs.inject_audio(SilenceSegment(5), 50);
        while(cs.microblocks_remaining()) {
            ks.stage_random_move();
            // Add the new node
            g.add_to_stack(new KlotskiBoard(ks.copy_board()));
            g.add_missing_edges(true);
            // Highlight the node of the board on the state-space graph
            gs.next_hash = ks.copy_board().get_hash();
            cs.render();
        }

        // Expand the graph until all nodes are present
        cs.inject_audio(SilenceSegment(2), (g2.size()-g.size())*1.2);
        while(cs.microblocks_remaining()) {
            g.expand_once();
            cs.render();
        }

        //Zoom in by moving to half the distance
        gs.state_manager.microblock_transition({
            {"d", ".1"},
        });
        cs.inject_audio_and_render(SilenceSegment(1));

        // Move the agent around on the board/graph
        cs.inject_audio(SilenceSegment(2), 5);
        while(cs.microblocks_remaining()) {
            ks.stage_random_move();
            gs.next_hash = ks.copy_board().get_hash();
            cs.render();
        }
    }
}

void render_video(){
    demo(); return;
    const double boardsize = 0.3;
    KlotskiBoard kb = intermediate;
    KlotskiScene ks(kb.w, kb.h, kb.representation, kb.rushhour, boardsize, boardsize*VIDEO_WIDTH/VIDEO_HEIGHT);
    CompositeScene cs;

    // Set up the graph scene with the current board state.
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks.copy_board()));
    GraphScene gs(&g);
    gs.state_manager.set({
        {"q1", "1"},
        {"qi", "<t> .1 * cos"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"},
        {"d", "1"},
        {"physics_multiplier","1"},
        {"attract","1"},
        {"repel","1"},
        {"decay",".8"},
        {"surfaces_opacity","0"},
        {"points_opacity","1"},
        {"lines_opacity","1"},
    });
    cs.add_scene(&gs, "gs");

    cs.inject_audio(FileSegment("What you're looking at is a random agent exploring the state-space graph of a slidy puzzle."), 50);
    while(cs.microblocks_remaining()) {
        ks.stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g.add_missing_edges(true);
        // Highlight the node of the board on the state-space graph
        gs.next_hash = ks.copy_board().get_hash();
        cs.render();
        cout << "AAAA" << endl;
    }
    cs.add_scene_fade_in(&ks, "ks", boardsize*.5, boardsize*.5*VIDEO_WIDTH/VIDEO_HEIGHT);

    ks.highlight_char = 'b';
    cs.inject_audio_and_render(FileSegment("It wants to free this block from the hole on the right side of the board."));

    // Transition: show the current board as a single node.
    gs.state_manager.microblock_transition({
        {"surfaces_opacity",".5"}
    });
    cs.inject_audio_and_render(FileSegment("We'll represent the current position of the puzzle as a single node."));


    ks.stage_random_move();
    g.add_to_stack(new KlotskiBoard(ks.copy_board()));
    g.add_missing_edges(true);
    gs.next_hash = ks.copy_board().get_hash();
    cs.inject_audio_and_render(FileSegment("If we make a single move on the puzzle, we arrive at a different node."));

    cs.inject_audio_and_render(FileSegment("Since these two positions are only separated by a single move,"));

    cs.inject_audio_and_render(FileSegment("We draw an edge connecting their two nodes."));

    cs.inject_audio(FileSegment("Each node is connected to a few more, and drawing them, we construct this labyrinth of paths."), 5);
    // Simulate additional moves to build out the graph.
    while(cs.microblocks_remaining()) {
        ks.stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g.add_missing_edges(true);
        gs.next_hash = ks.copy_board().get_hash();
        cs.render();
    }

    cs.inject_audio(FileSegment("If we grow out the graph in its entirety, we can see that the random agent is still quite far away from the solutions."), 10);
    while(cs.microblocks_remaining()) {
        ks.stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        g.add_missing_edges(true);
        gs.next_hash = ks.copy_board().get_hash();
        cs.render();
    }

    cs.inject_audio_and_render(FileSegment("We can also make a lot more observations about the topology of this puzzle!"));

    cs.inject_audio_and_render(FileSegment("But, let's start simple."));

    cs.inject_audio_and_render(FileSegment("This is a beginner's puzzle."));
}
