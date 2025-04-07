#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Media/LoopAnimationScene.cpp"

// Lists of KlotskiBoards
auto gpt = {gpt2, gpt3, weird1, weird2, weird3};
auto other = {apk, mathgames_12_13_04, euler766_easy};
auto suns = {fatsun, sun_no_minis, truncatedsun};
auto rushhours = {beginner, intermediate, advanced, expert, reddit, guh3, guh4, video, thinkfun1, thinkfun2, thinkfun3};
auto big = {sun};
auto geometry = {jam3x3, cube_4d, cube_6d, big_block, diamond, doublering, outer_ring, plus_3_corners, plus_4_corners, ring, ring_big, rows, small_block, t_shapes, triangles, triangles2, manifold_2d, manifold_3d};

int get_graph_size(const KlotskiBoard& kb){
    Graph g;
    g.add_to_stack(new KlotskiBoard(kb));
    g.expand_completely();
    return g.size();
}

void part1(){
    CompositeScene cs;

    // Add KlotskiScene and GraphScene for Intermediate puzzle, begin making 50 random moves.
    KlotskiScene ks(intermediate);
    StateSet board_width_height{{"w",".3"},{"h",to_string(.3*VIDEO_WIDTH/VIDEO_HEIGHT)}};
    StateSet board_position    {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}};
    ks.state_manager.set(board_width_height);
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks.copy_board()));
    GraphScene gs(&g);
    StateSet default_graph_state{
        {"q1", "1"},
        {"qi", "<t> .1 * cos"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"}, // Camera orientation quaternion
        {"decay",".8"},
        {"surfaces_opacity","0"}, // Whether we want to draw the board at every node
        {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
    };
    gs.state_manager.set(default_graph_state);
    cs.add_scene(&gs, "gs");
    cs.add_scene(&ks, "ks");
    cs.state_manager.set(board_position);
    cs.stage_macroblock(FileSegment("What you're looking at is a random agent exploring the state-space graph of a slidy puzzle."), 200);
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
    cs.stage_macroblock(FileSegment("It wants to free this block from the hole on the right side of the board."), 4);
    cs.state_manager.microblock_transition({{"ks.x",".5"},{"ks.y",".5"}});
    ks.state_manager.microblock_transition({{"w","1"},{"h","1"}});
    cs.render();
    ks.highlight_char = 'b';
    cs.render();
    int shift_dist = 4;
    ks.stage_move('b', shift_dist, 0);
    cs.render();
    ks.stage_move('b', -shift_dist, 0);
    cs.render();

    ks.highlight_char = 'a';
    cs.stage_macroblock_and_render(FileSegment("However, it can't do that yet, since this piece is in the way..."));

    cs.state_manager.microblock_transition(board_position);
    ks.state_manager.microblock_transition(board_width_height);
    cs.stage_macroblock_and_render(SilenceSegment(1));

    // Delete all nodes of the graph except for the current one. Turn on surface opacity, and turn off edge opacity.
    cs.stage_macroblock(SilenceSegment(3), g.size()-1);
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
    cs.stage_macroblock_and_render(FileSegment("We'll represent the current position of the puzzle as a single node."));

    // Make one move and insert it on the graph.
    ks.highlight_char = 'c';
    cs.stage_macroblock(FileSegment("If we make a single move on the puzzle,"), 5);
    ks.stage_move('c', 0, 1);
    cs.render();
    ks.stage_move('c', 0, -1);
    cs.render();
    ks.stage_move('c', 0, 1);
    cs.render();
    ks.stage_move('c', 0, -1);
    cs.render();
    ks.stage_move('c', 0, 1);
    cs.render();
    g.add_to_stack(new KlotskiBoard(ks.copy_staged_board()));
    g.add_missing_edges(true);
    gs.next_hash = ks.copy_board().get_hash();
    cs.stage_macroblock_and_render(FileSegment("we arrive at a different node."));

    // TODO not yet animated
    cs.stage_macroblock_and_render(FileSegment("Since these two positions are only separated by a single move,"));

    // Turn edge opacity on.
    gs.state_manager.macroblock_transition({{"lines_opacity","1"}});
    cs.stage_macroblock_and_render(FileSegment("We draw an edge connecting their two nodes."));

    // Make a few random moves.
    gs.state_manager.macroblock_transition({{"surfaces_opacity","0"}});
    cs.stage_macroblock(FileSegment("Each node is connected to a few more, and drawing them, we construct this labyrinth of paths."), 8);
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
    cs.stage_macroblock_and_render(SilenceSegment(2));

    // Expand the graph by one node until it is halfway complete. Fade out everything from the CompositeScene and then delete scenes when faded out.
    cs.fade_out_all_scenes(false);
    cs.stage_macroblock(FileSegment("If we make all of the possible moves, what do you think the graph would look like?"), get_graph_size(intermediate) * .7);
    while(cs.microblocks_remaining()) {
        g.expand_once();
        cs.render();
    }
    cs.remove_all_scenes();

    // Create new GraphScene (with no corresponding KlotskiScene) for manifold_2d, fade it in while expanding the graph completely
    cs.stage_macroblock(FileSegment("Maybe there would be some overarching structure..."), get_graph_size(manifold_2d));
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    GraphScene gs2d(&g2d);
    gs2d.state_manager.set(default_graph_state);
    cs.add_scene_fade_in(&gs2d, "gs2d");
    while(cs.microblocks_remaining()){
        g2d.expand_once();
        cs.render();
    }

    // Transition the 2D grid scene to the left (by setting its width to .5 and moving its x position to .25)
    StateSet grid_transition{{"w","0.5"},{"x","0.25"}};
    gs2d.state_manager.microblock_transition(grid_transition);
      cs.state_manager.microblock_transition({{"gs2d.x",".25"}});
    cs.stage_macroblock_and_render(FileSegment("such as a two-dimensional grid,"));

    // Create new GraphScene for manifold_3d on the right side of the screen and fade it in while expanding the graph completely 
    cs.stage_macroblock(FileSegment("or a three-dimensional crystal lattice!"), get_graph_size(manifold_3d));
    Graph g3d;
    g3d.add_to_stack(new KlotskiBoard(manifold_3d));
    GraphScene gs3d(&g3d, .5, 1);
    gs3d.state_manager.set(default_graph_state);
    cs.add_scene_fade_in(&gs3d, "gs3d", 0.75, 0.5);
    while(cs.microblocks_remaining()){
        g3d.expand_once();
        cs.render();
    }

    // Fade out all scenes and then delete them
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(SilenceSegment(1));
    cs.remove_all_scenes();

    // Fade in and expand a GraphScene for intermediate again, but this time override "physics_multiplier" to be zero so the graph structure is indiscernable.
    cs.stage_macroblock(FileSegment("Maybe it's an incomprehensibly dense mesh of interconnected nodes with no grand structure."), 100);
    Graph g_int;
    g_int.add_to_stack(new KlotskiBoard(intermediate));
    GraphScene gs_int(&g_int);
    gs_int.state_manager.set(default_graph_state);
    gs_int.state_manager.set({{"attract","-1"}, {"repel","-1"}});
    cs.add_scene_fade_in(&gs_int, "gs_int");
    while(cs.microblocks_remaining()){
        g_int.expand_once();
        cs.render();
    }

    // Fade out all scenes and then delete them
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(SilenceSegment(1));
    cs.remove_all_scenes();
}

void part2() {
    // Start over by adding a KlotskiScene.
    KlotskiScene ks(sun);

    // TODO Make moves according to the shortest path to the position given by "abbcabbc.gehj.ehddif"
    KlotskiBoard transition1(4, 5, "abbcabbc.gehj.ehddif", false);
    ks.stage_macroblock_and_render(FileSegment("I fell down this rabbit hole when I was shown this particular slidy puzzle."));

    // TODO Make moves following the shortest path to the position given by "abbcabbcfidde.gheijh"
    KlotskiBoard transition2(4, 5, "abbcabbcfidde.gheijh", false);
    ks.stage_macroblock_and_render(FileSegment("It's called Klotski, and it's quite old."));

    // Hotswap to a new KlotskiScene "ks2" with only the sun on it.
    KlotskiScene ks2(KlotskiBoard(4, 5, ".bb..bb.............", false));
    // Show piece 'b' getting moved 3 units downward and back.
    ks2.stage_macroblock_and_render(FileSegment("The goal is to get this big piece out of the bottom."));

    // Swap back to ks. Transition "dots" state element to 1 and back to 0 to introduce the pins over two microblocks
    ks.stage_macroblock(FileSegment("This one is slightly different in that there are no pins in the intersections,"), 2);
    ks.state_manager.microblock_transition({{"dots", "1"}});
    ks.render();
    ks.state_manager.microblock_transition({{"dots", "0"}});
    ks.render();

    // Now that dots are back to 0, demonstrate a lateral move. (move piece 'e' right one space)
    ks.stage_move('e', 1, 0);
    ks.stage_macroblock_and_render(FileSegment("meaning blocks are free to move laterally."));

    // TODO Show the intermediate puzzle from before.
    CompositeScene cs;
    cs.add_scene(&ks, "ks");
    KlotskiScene ks_intermediate(intermediate);
    cs.add_scene_fade_in(&ks_intermediate, "ks_intermediate", .75, .5);
    cs.state_manager.microblock_transition({{"ks.x",".25"}});
    ks.state_manager.microblock_transition({{"w",".5"}});
    ks.stage_macroblock_and_render(FileSegment("Compared to the previous puzzle, it's _much harder_."));

    // Looping animation scene - me and coworker
    LoopAnimationScene las({"coworker1", "coworker2", "coworker3", "give1", "give2", "give3", "trying1", "trying2", "trying3", "solved1", "solved2", "solved3", "dizzy1", "dizzy2", "dizzy3"});
    las.state_manager.set({{"loop_length", "3"}});
    las.stage_macroblock(FileSegment("I showed this one to a coworker before heading home,"), 2);
    las.render();
    las.state_manager.set({{"loop_start", "3"}});
    las.render();

    // Looping animation scene - coworker struggling to solve
    las.state_manager.set({{"loop_start", "6"}});
    las.stage_macroblock_and_render(FileSegment("but without telling me, he decided to stay until it was solved."));

    // Looping animation scene - coworker dizzy, puzzle solved
    las.stage_macroblock(FileSegment("He told me he finally got it... around 11:00 PM."), 2);
    las.state_manager.set({{"loop_start", "9"}});
    las.render();
    las.state_manager.set({{"loop_start", "12"}});
    las.render();

    // Transition to subpuzzle containing only blocks b and d
    cs.state_manager.set({{"ks.x",".5"}});
    ks.state_manager.set({{"w","1"}});
    cs.fade_out_all_scenes();
    KlotskiScene ks_bd(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene_fade_in(&ks_bd, "ks_bd");
    cs.stage_macroblock_and_render(FileSegment("His conjecture was that the hardest part of the puzzle was moving the big piece underneath this horizontal block."));

    ks.stage_macroblock_and_render(FileSegment("To be honest, I still haven't even bothered solving it myself..."));
    ks.stage_macroblock_and_render(FileSegment("I became more interested in seeing what this thing really looks like under the hood."));
}

void part3() {
    CompositeScene cs;
    cs.stage_macroblock_and_render(FileSegment("Step 1 is the easy part."));
    cs.stage_macroblock_and_render(FileSegment("Just make every possible move on the puzzle, and keep track of which states are reachable from which other states."));
    cs.stage_macroblock_and_render(FileSegment("The hard part is representing this mess of data visually."));
    cs.stage_macroblock_and_render(FileSegment("Now, there are lots of papers about how to do just that,"));
    cs.stage_macroblock_and_render(FileSegment("but papers are for party poopers!"));
    cs.stage_macroblock_and_render(FileSegment("Think of it like this... if it only takes one move to get from position A to position B, we want those two nodes to be close to each other."));
    cs.stage_macroblock_and_render(FileSegment("If it takes longer, we want them to be further away."));
    cs.stage_macroblock_and_render(FileSegment("So, how about we just make nearby nodes pull each other closer, as though there was some rubber band tying them together?"));
    cs.stage_macroblock_and_render(FileSegment("Well, without any force to counteract it, they will all just get sucked together."));
    cs.stage_macroblock_and_render(FileSegment("So, we will have non-neighbor nodes repel each other."));
    cs.stage_macroblock_and_render(FileSegment("In other words, this whole graph is basically just simulated physically, where the nodes are protons and the edges are springs."));
}

void render_video() {
    //part1();
    part2();
    //part3();
    // TODO Add a section analyzing the terrain of the final klotski graph
}
