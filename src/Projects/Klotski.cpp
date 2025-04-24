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
    auto ks_ptr = make_shared<KlotskiScene>(intermediate);
    StateSet board_width_height{{"w",".3"},{"h",to_string(.3*VIDEO_WIDTH/VIDEO_HEIGHT)}};
    StateSet board_position    {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}};
    ks_ptr->state_manager.set(board_width_height);
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    auto gs_ptr = std::make_shared<GraphScene>(&g);
    StateSet default_graph_state{
        {"q1", "1"},
        {"qi", "<t> .1 * cos"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"}, // Camera orientation quaternion
        {"decay",".8"},
        {"surfaces_opacity","0"}, // Whether we want to draw the board at every node
        {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
    };
    gs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(gs_ptr, "gs");
    cs.add_scene(ks_ptr, "ks");
    cs.state_manager.set(board_position);
    cs.stage_macroblock(FileSegment("What you're looking at is a random agent exploring the state-space graph of a slidy puzzle."), 200);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
        g.add_missing_edges(true);
        // Highlight the node of the board on the state-space graph
        gs_ptr->next_hash = ks_ptr->copy_board().get_hash();
        cs.render_microblock(); // Render a microblock
    }

    // Set ks.highlight_char to highlight the block which needs to get freed.
    cs.stage_macroblock(FileSegment("It wants to free this block from the hole on the right side of the board."), 4);
    cs.state_manager.microblock_transition({{"ks.x",".5"},{"ks.y",".5"}});
    ks_ptr->state_manager.microblock_transition({{"w","1"},{"h","1"}});
    cs.render_microblock();
    ks_ptr->highlight_char = 'b';
    cs.render_microblock();
    int shift_dist = 4;
    ks_ptr->stage_move({'b', shift_dist, 0});
    cs.render_microblock();
    ks_ptr->stage_move({'b', -shift_dist, 0});
    cs.render_microblock();

    ks_ptr->highlight_char = 'a';
    cs.stage_macroblock(FileSegment("However, it can't do that yet, since this piece is in the way..."), 1);
    cs.render_microblock();

    cs.state_manager.microblock_transition(board_position);
    ks_ptr->state_manager.microblock_transition(board_width_height);
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();

    // Delete all nodes of the graph except for the current one. Turn on surface opacity, and turn off edge opacity.
    cs.stage_macroblock(SilenceSegment(3), g.size()-1);
    double kb_to_keep = ks_ptr->copy_board().get_hash();
    gs_ptr->state_manager.macroblock_transition({{"surfaces_opacity",".5"},{"lines_opacity","0"}});
    gs_ptr->state_manager.set({{"centering_strength","0"}});
    unordered_map<double,Node> nodes_copy = g.nodes;
    for(auto it = nodes_copy.begin(); it != nodes_copy.end(); ++it){
        double id_here = it->first;
        if(id_here == kb_to_keep) continue;
        g.remove_node(id_here);
        cs.render_microblock();
    }
    g.clear_queue();

    // Re-center
    gs_ptr->state_manager.microblock_transition({{"centering_strength","1"}});
    cs.stage_macroblock(FileSegment("We'll represent the current position of the puzzle as a single node."), 1);
    cs.render_microblock();

    // Make one move and insert it on the graph.
    ks_ptr->highlight_char = 'c';
    cs.stage_macroblock(FileSegment("If we make a single move on the puzzle,"), 5);
    ks_ptr->stage_move({'c', 0, 1});
    cs.render_microblock();
    ks_ptr->stage_move({'c', 0, -1});
    cs.render_microblock();
    ks_ptr->stage_move({'c', 0, 1});
    cs.render_microblock();
    ks_ptr->stage_move({'c', 0, -1});
    cs.render_microblock();
    ks_ptr->stage_move({'c', 0, 1});
    cs.render_microblock();
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
    g.add_missing_edges(true);
    gs_ptr->next_hash = ks_ptr->copy_board().get_hash();
    cs.stage_macroblock(FileSegment("we arrive at a different node."), 1);
    cs.render_microblock();

    // TODO not yet animated
    cs.stage_macroblock(FileSegment("Since these two positions are only separated by a single move,"), 1);
    cs.render_microblock();

    // Turn edge opacity on.
    gs_ptr->state_manager.macroblock_transition({{"lines_opacity","1"}});
    cs.stage_macroblock(FileSegment("We draw an edge connecting their two nodes."), 1);
    cs.render_microblock();

    // Make a few random moves.
    gs_ptr->state_manager.macroblock_transition({{"surfaces_opacity","0"}});
    cs.stage_macroblock(FileSegment("Each node is connected to a few more, and drawing them, we construct this labyrinth of paths."), 8);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
        g.add_missing_edges(true);
        gs_ptr->next_hash = ks_ptr->copy_board().get_hash();
        cs.render_microblock();
    }

    // Unhighlight and fade out the KlotskiScene
    gs_ptr->next_hash = 0;
    cs.state_manager.macroblock_transition({{"ks.opacity","0"}});
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();

    // Expand the graph by one node until it is halfway complete. Fade out everything from the CompositeScene and then delete scenes when faded out.
    cs.fade_out_all_scenes(false);
    cs.stage_macroblock(FileSegment("If we make all of the possible moves, what do you think the graph would look like?"), get_graph_size(intermediate) * .7);
    while(cs.microblocks_remaining()) {
        g.expand_once();
        cs.render_microblock();
    }
    cs.remove_all_scenes();

    // Create new GraphScene (with no corresponding KlotskiScene) for manifold_2d, fade it in while expanding the graph completely
    cs.stage_macroblock(FileSegment("Maybe there would be some overarching structure..."), get_graph_size(manifold_2d));
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    auto gs2d_ptr = make_shared<GraphScene>(&g2d);
    gs2d_ptr->state_manager.set(default_graph_state);
    cs.add_scene_fade_in(gs2d_ptr, "gs2d");
    while(cs.microblocks_remaining()){
        g2d.expand_once();
        cs.render_microblock();
    }

    // Transition the 2D grid scene to the left (by setting its width to .5 and moving its x position to .25)
    StateSet grid_transition{{"w","0.5"},{"x","0.25"}};
    gs2d_ptr->state_manager.microblock_transition(grid_transition);
    cs.state_manager.microblock_transition({{"gs2d.x",".25"}});
    cs.stage_macroblock(FileSegment("such as a two-dimensional grid,"), 1);
    cs.render_microblock();

    // Create new GraphScene for manifold_3d on the right side of the screen and fade it in while expanding the graph completely 
    cs.stage_macroblock(FileSegment("or a three-dimensional crystal lattice!"), get_graph_size(manifold_3d));
    Graph g3d;
    g3d.add_to_stack(new KlotskiBoard(manifold_3d));
    auto gs3d_ptr = make_shared<GraphScene>(&g3d, .5, 1);
    gs3d_ptr->state_manager.set(default_graph_state);
    cs.add_scene_fade_in(gs3d_ptr, "gs3d", 0.75, 0.5);
    while(cs.microblocks_remaining()){
        g3d.expand_once();
        cs.render_microblock();
    }

    // Fade out all scenes and then delete them
    cout << "A" << endl;
    cs.fade_out_all_scenes();
    cout << "B" << endl;
    cs.stage_macroblock(SilenceSegment(1), 1);
    cout << "C" << endl;
    cs.render_microblock();
    cout << "D" << endl;
    cs.remove_all_scenes();

    // Fade in and expand a GraphScene for intermediate again, but this time override "physics_multiplier" to be zero so the graph structure is indiscernable.
    cs.stage_macroblock(FileSegment("Maybe it's an incomprehensibly dense mesh of interconnected nodes with no grand structure."), 100);
    Graph g_int;
    g_int.add_to_stack(new KlotskiBoard(intermediate));
    auto gs_int_ptr = make_shared<GraphScene>(&g_int);
    gs_int_ptr->state_manager.set(default_graph_state);
    gs_int_ptr->state_manager.set({{"attract","-1"}, {"repel","-1"}});
    cs.add_scene_fade_in(gs_int_ptr, "gs_int");
    while(cs.microblocks_remaining()){
        g_int.expand_once();
        cs.render_microblock();
    }

    // Fade out all scenes and then delete them
    cs.fade_out_all_scenes();
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    cs.remove_all_scenes();
}

void part2() {
    // Start over by adding a KlotskiScene.
    shared_ptr<KlotskiScene> ks_ptr = make_shared<KlotskiScene>(sun);

    // Abstracted function for performing shortest path moves.
    auto perform_shortest_path = [&](KlotskiBoard end, const string &msg) {
        Graph g;
        g.add_to_stack(new KlotskiBoard(sun));
        g.expand_completely();
        auto path = g.shortest_path(ks_ptr->copy_board().get_hash(), end.get_hash()).first;
        ks_ptr->stage_macroblock(FileSegment(msg), path.size()-1);
        path.pop_front();
        while(ks_ptr->microblocks_remaining()){
            double next = *(path.begin());
            path.pop_front();
            Node node = g.nodes.at(next);
            KlotskiBoard* next_board = dynamic_cast<KlotskiBoard*>(node.data);
            ks_ptr->stage_move(ks_ptr->copy_board().move_required_to_reach(*next_board));
            ks_ptr->render_microblock();
        }
    };

    // Make moves according to the shortest path to the position given
    perform_shortest_path(KlotskiBoard(4, 5, "abbcabbc.gehj.ehddif", false),
                          "I fell down this rabbit hole when I was shown this particular slidy puzzle.");

    // Make moves following the shortest path to the position given
    perform_shortest_path(KlotskiBoard(4, 5, "abbcabbcfidde.ghe.jh", false),
                          "It's called Klotski, and it's quite old.");

    // Hotswap to a new KlotskiScene "ks2" with only the sun on it.
    auto ks2_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb.............", false));
    // Show piece 'b' getting moved 3 units downward and back.
    ks2_ptr->stage_macroblock(FileSegment("The goal is to get this big piece out of the bottom."), 3);
    ks2_ptr->stage_move({'b', 0, 3});
    ks2_ptr->render_microblock();
    ks2_ptr->stage_move({'b', 0, -3});
    ks2_ptr->render_microblock();

    // Swap back to ks. Transition "dots" state element to 1 and back to 0 to introduce the pins over two microblocks
    ks_ptr->stage_macroblock(FileSegment("This one is slightly different in that there are no pins in the intersections,"), 2);
    ks_ptr->state_manager.microblock_transition({{"dots", "1"}});
    ks_ptr->render_microblock();
    ks_ptr->state_manager.microblock_transition({{"dots", "0"}});
    ks_ptr->render_microblock();

    // Now that dots are back to 0, demonstrate a lateral move. (move piece 'e' right one space)
    ks_ptr->stage_move({'e', 1, 0});
    ks_ptr->stage_macroblock(FileSegment("meaning blocks are free to move laterally."), 1);
    ks_ptr->render_microblock();

    // Show the intermediate puzzle from before.
    CompositeScene cs;
    cs.add_scene(ks_ptr, "ks");
    cs.add_scene_fade_in(make_shared<KlotskiScene>(intermediate, .5, 1), "ks_intermediate", .25, .5);
    cs.state_manager.microblock_transition({{"ks.x",".75"}});
    ks_ptr->state_manager.microblock_transition({{"w",".5"}});
    cs.stage_macroblock(FileSegment("Compared to the previous puzzle, it's _much harder_."), 1);
    cs.render_microblock();

    // Looping animation scene - me and coworker
    LoopAnimationScene las({"coworker1", "coworker2", "coworker3", 
                            "give1", "give2", "give3",
                            "trying1", "trying2", "trying3",
                            "solved1", "solved2", "solved3",
                            "dizzy1", "dizzy2", "dizzy3"});
    las.state_manager.set({{"loop_length", "3"}});
    las.stage_macroblock(FileSegment("I showed this one to a coworker before heading home,"), 2);
    las.render_microblock();
    las.state_manager.set({{"loop_start", "3"}});
    las.render_microblock();

    // Looping animation scene - coworker struggling to solve
    las.state_manager.set({{"loop_start", "6"}});
    las.stage_macroblock(FileSegment("but without telling me, he decided to stay until it was solved."), 1);
    las.render_microblock();

    // Looping animation scene - coworker dizzy, puzzle solved
    las.stage_macroblock(FileSegment("He told me he finally got it... around 11:00 PM."), 2);
    las.state_manager.set({{"loop_start", "9"}});
    las.render_microblock();
    las.state_manager.set({{"loop_start", "12"}});
    las.render_microblock();

    // Transition to subpuzzle containing only blocks b and d
    cs.remove_all_scenes();
    ks_ptr = make_shared<KlotskiScene>(sun);
    cs.add_scene(ks_ptr, "ks");
    cs.fade_out_all_scenes();
    auto ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene_fade_in(ks_bd_ptr, "ks_bd");
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    cs.remove_scene(ks_ptr);

    // Animate big piece going under small piece.
    cs.stage_macroblock(FileSegment("His conjecture was that the hardest part of the puzzle was moving the big piece underneath this horizontal block."), 5);
    ks_bd_ptr->stage_move({'b', 1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'d', -1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', 0, 3});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'d', 1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', -1, 0});
    cs.render_microblock();

    // Fade ks back in from ks_bd.
    cs.fade_out_all_scenes();
    cs.add_scene_fade_in(ks_ptr, "ks", 0.5, 0.5);
    cs.stage_macroblock(FileSegment("To be honest, I still haven't even bothered solving it myself..."), 1);
    cs.render_microblock();
    cs.render_microblock();

    // Start to grow a graph (a hundred nodes or so) in the background
    cs.stage_macroblock(FileSegment("I became more interested in seeing what this thing really looks like under the hood."), 100);
    Graph bg_graph;
    bg_graph.add_to_stack(new KlotskiBoard(sun));
    auto bggs_ptr = make_shared<GraphScene>(&bg_graph);
    bggs_ptr->state_manager.set({{"surfaces_opacity", "0.3"}, {"lines_opacity", "0.5"}});
    cs.add_scene(bggs_ptr, "bggs");
    while(cs.microblocks_remaining()){
        bg_graph.expand_once();
        cs.render_microblock();
    }
}

void render_video() {
    FOR_REAL = false;
    //PRINT_TO_TERMINAL = false;
    part1();
    part2();
}
