#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/PauseScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Media/LoopAnimationScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"

// Lists of KlotskiBoards
auto gpt = {gpt2, gpt3, weird1, weird2, weird3};
auto other = {apk, mathgames_12_13_04, euler766_easy};
auto suns = {fatsun, sun_no_minis, truncatedsun};
auto rushhours = {beginner, intermediate, advanced, expert, reddit, guh3, guh4, video, thinkfun1, thinkfun2, thinkfun3};
auto big = {sun};
auto geometry = {jam3x3, cube_4d, cube_6d, big_block, diamond, doublering, outer_ring, plus_3_corners, plus_4_corners, ring, ring_big, rows, small_block, t_shapes, triangles, triangles2, manifold_1d, manifold_2d, manifold_3d, manifold_4d, full_15_puzzle};

int get_graph_size(const KlotskiBoard& kb){
    Graph g;
    g.add_to_stack(new KlotskiBoard(kb));
    g.expand_completely();
    return g.size();
}

StateSet default_graph_state{
    {"q1", "1"},
    {"qi", "<t> .2 * cos"},
    {"qj", "<t> .314 * sin"},
    {"qk", "0"}, // Camera orientation quaternion
    {"decay",".8"},
    {"surfaces_opacity","0"}, // Whether we want to draw the board at every node
    {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
};
StateSet spinny{
    {"qi", "<t> .8 * cos"},
    {"qj", "<t> .6 * sin"},
    {"qk", "<t> .9 * sin"},
};
StateSet board_width_height{{"w",".3"},{"h",to_string(.3*VIDEO_WIDTH/VIDEO_HEIGHT)}};
StateSet board_position    {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}};
double yval = .15*VIDEO_WIDTH/VIDEO_HEIGHT;

void perform_shortest_path(shared_ptr<KlotskiScene> ks_ptr, KlotskiBoard end, const string &msg) {
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
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

void part1(){
    CompositeScene cs;

    // Add KlotskiScene and GraphScene for Intermediate puzzle, begin making 50 random moves.
    auto ks_ptr = make_shared<KlotskiScene>(intermediate);
    ks_ptr->state_manager.set(board_width_height);
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    auto gs_ptr = std::make_shared<GraphScene>(&g);
    gs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(gs_ptr, "gs");
    cs.add_scene(ks_ptr, "ks");
    cs.state_manager.set(board_position);
    cs.stage_macroblock(FileSegment("You're looking at a random agent exploring the state-space graph of a slidy puzzle."), 100);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges(true);
        // Highlight the node of the board on the state-space graph
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Set ks.highlight_char to highlight the block which needs to get freed.
    auto ksb_ptr= make_shared<KlotskiScene>(KlotskiBoard(6, 6, "............bb......................", true));
    ksb_ptr->state_manager.set(board_width_height);
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.add_scene_fade_in(ksb_ptr, "ksb");
    cs.state_manager.set({{"ksb.x",".15"},{"ksb.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.fade_subscene("ks", 0);
    cs.state_manager.microblock_transition({{"ks.x",".5"},{"ks.y",".5"}});
    cs.state_manager.microblock_transition({{"ksb.x",".5"},{"ksb.y",".5"}});
    cs.fade_subscene("gs", 0);
    ks_ptr->state_manager.microblock_transition({{"w","1"},{"h","1"}});
    ksb_ptr->state_manager.microblock_transition({{"w","1"},{"h","1"}});
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("It wants to free this block from the hole on the right side."), 3);
    cs.render_microblock();
    ksb_ptr->highlight_char = 'b';
    cs.render_microblock();
    ksb_ptr->stage_move({'b', 12, 0});
    cs.render_microblock();

    cs.fade_subscene("ksb", 0);
    cs.fade_subscene("ks", 1);
    ks_ptr->highlight_char = 'a';
    cs.stage_macroblock(FileSegment("However, it can't do that yet, since this piece is in the way..."), 1);
    cs.render_microblock();

    cs.fade_subscene("gs", 1);
    cs.state_manager.microblock_transition(board_position);
    ks_ptr->state_manager.microblock_transition(board_width_height);
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();

    // Delete all nodes of the graph except for the current one. Turn on surface opacity, and turn off edge opacity.
    cs.stage_macroblock(SilenceSegment(3), g.size()-1);
    gs_ptr->state_manager.macroblock_transition({{"surfaces_opacity",".5"},{"lines_opacity","0"}});
    gs_ptr->state_manager.set({{"centering_strength","0"}});
    unordered_map<double,Node> nodes_copy = g.nodes;
    for(auto it = nodes_copy.begin(); it != nodes_copy.end(); ++it){
        double id_here = it->first;
        if(id_here == gs_ptr->curr_hash) continue;
        g.remove_node(id_here);
        cs.render_microblock();
    }
    g.clear_queue();

    // Re-center
    gs_ptr->state_manager.microblock_transition({{"centering_strength","0.1"}});
    cs.stage_macroblock(FileSegment("We'll represent the current position of the puzzle as a node."), 1);
    cs.render_microblock();

    // Make one move and insert it on the graph.
    ks_ptr->highlight_char = 'c';
    gs_ptr->state_manager.microblock_transition({{"centering_strength","1"}});
    gs_ptr->state_manager.microblock_transition({{"physics_multiplier","1"}});
    cs.stage_macroblock(FileSegment("If we make one move on the puzzle,"), 5);
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
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.stage_macroblock(FileSegment("we arrive at a different node."), 1);
    cs.render_microblock();
    gs_ptr->state_manager.microblock_transition({{"physics_multiplier","1"}});

    cs.stage_macroblock(FileSegment("Since these positions are just one move apart,"), 1);
    cs.render_microblock();

    // Turn edge opacity on.
    gs_ptr->state_manager.macroblock_transition({{"lines_opacity","1"}});
    gs_ptr->state_manager.microblock_transition({{"physics_multiplier","5"}});
    cs.stage_macroblock(FileSegment("We draw an edge connecting the two nodes."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceSegment(1.5), 2);
    ks_ptr->stage_move({'c', 0, -1});
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_ptr->stage_move({'c', 0, 1});
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.render_microblock();

    // Make a few random moves.
    gs_ptr->state_manager.macroblock_transition({{"surfaces_opacity","0"}});
    cs.stage_macroblock(FileSegment("Each node is connected to a few more, and drawing them, we construct this labyrinth of paths."), 8);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges(true);
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Unhighlight and fade out the KlotskiScene
    gs_ptr->next_hash = 0;
    cs.state_manager.macroblock_transition({{"ks.opacity","-2"}});
    cs.state_manager.macroblock_transition({{"gs.opacity","0"}});
    // Expand the graph by one node until it is halfway complete. Fade out everything from the CompositeScene and then delete scenes when faded out.
    cs.stage_macroblock(FileSegment("You might start to wonder- if we add all the nodes, what would the graph look like?"), get_graph_size(intermediate) * .7);
    while(cs.microblocks_remaining()) {
        g.expand_once();
        cs.render_microblock();
    }
    cs.remove_all_subscenes();
}

void part2() {
    CompositeScene cs;

    Graph g2d;
    {
        // Create new GraphScene (with no corresponding KlotskiScene) for manifold_2d, fade it in while expanding the graph completely
        g2d.add_to_stack(new KlotskiBoard(manifold_2d));
        auto gs2d_ptr = make_shared<GraphScene>(&g2d);
        gs2d_ptr->state_manager.set(default_graph_state);
        cs.add_scene(gs2d_ptr, "gs2d");
        cs.stage_macroblock(FileSegment("Maybe it has some overarching structure..."), get_graph_size(manifold_2d));
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
    }

    Graph g3d;
    {
        // Create new GraphScene for manifold_3d on the right side of the screen and fade it in while expanding the graph completely 
        cs.stage_macroblock(FileSegment("or a 3d crystal lattice!"), get_graph_size(manifold_3d));
        g3d.add_to_stack(new KlotskiBoard(manifold_3d));
        auto gs3d_ptr = make_shared<GraphScene>(&g3d, .5, 1);
        gs3d_ptr->state_manager.set(default_graph_state);
        cs.add_scene(gs3d_ptr, "gs3d", 0.75, 0.5);
        while(cs.microblocks_remaining()){
            g3d.expand_once();
            cs.render_microblock();
        }

        // Fade out all scenes and then delete them
        cs.fade_all_subscenes(0);
        cs.stage_macroblock(SilenceSegment(1), 1);
        cs.render_microblock();
        cs.remove_all_subscenes();
    }

    {
        // Fade in and expand a GraphScene for intermediate again, but this time override "physics_multiplier" to be zero so the graph structure is indiscernable.
        cs.stage_macroblock(FileSegment("Maybe it's a dense mesh of interconnected nodes with no grand structure."), 100);
        Graph g_int;
        g_int.add_to_stack(new KlotskiBoard(intermediate));
        auto gs_int_ptr = make_shared<GraphScene>(&g_int);
        gs_int_ptr->state_manager.set(default_graph_state);
        gs_int_ptr->state_manager.set({{"attract","0"}, {"repel","0"}});
        //gs_int_ptr->state_manager.set({{"attract","-1"}, {"repel","-1"}});
        cs.add_scene_fade_in(gs_int_ptr, "gs_int");
        while(cs.microblocks_remaining()){
            g_int.expand_once();
            cs.render_microblock();
        }

        // Fade out all scenes and then delete them
        cs.fade_all_subscenes(0);
        cs.stage_macroblock(SilenceSegment(1), 1);
        cs.render_microblock();
        cs.remove_all_subscenes();
    }
}

void part3() {
    // Start over by adding a KlotskiScene.
    shared_ptr<KlotskiScene> ks_ptr = make_shared<KlotskiScene>(sun);

    CompositeScene cs;
    cs.add_scene_fade_in(ks_ptr, "ks");
    cs.stage_macroblock(SilenceSegment(.8), 1);
    cs.render_microblock();

    // Make moves according to the shortest path to the position given
    perform_shortest_path(ks_ptr, KlotskiBoard(4, 5, "abbcabbc.gehj.ehddif", false), "I fell down this rabbit hole when I was shown this puzzle.");

    // Make moves following the shortest path to the position given
    perform_shortest_path(ks_ptr, KlotskiBoard(4, 5, "abbcabbcfidde.ghe.jh", false), "It's called Klotski.");

    // Hotswap to a new KlotskiScene "ks2" with only the sun on it.
    auto ks2_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb.............", false));
    // Show piece 'b' getting moved 3 units downward and back.
    cs.add_scene(ks2_ptr, "ks2");
    cs.stage_macroblock(FileSegment("The goal is to get this big piece out of the bottom."), 2);
    cs.fade_subscene("ks", 0);
    cs.render_microblock();
    ks2_ptr->stage_move({'b', 0, 8});
    cs.render_microblock();

    // Swap back to ks. Transition "dots" state element to 1 and back to 0 to introduce the pins over two microblocks
    cs.stage_macroblock(FileSegment("It's slightly different since the intersections don't have pins,"), 3);
    cs.state_manager.microblock_transition({{"ks.opacity","1"},});
    cs.render_microblock();
    ks_ptr->state_manager.microblock_transition({{"dots", "1"}});
    cs.render_microblock();
    ks_ptr->state_manager.microblock_transition({{"dots", "0"}});
    cs.render_microblock();

    // Now that dots are back to 0, demonstrate a lateral move. (move piece 'e' right one space)
    ks_ptr->stage_move({'e', 1, 0});
    ks_ptr->stage_macroblock(FileSegment("so blocks are free to move laterally."), 1);
    ks_ptr->render_microblock();

    // Show the intermediate puzzle from before.
    cs.remove_all_subscenes();
    cs.add_scene(ks_ptr, "ks");
    cs.add_scene_fade_in(make_shared<KlotskiScene>(intermediate, .5, 1), "ks_intermediate", .25, .5);
    cs.state_manager.microblock_transition({{"ks.x",".75"}});
    ks_ptr->state_manager.microblock_transition({{"w",".5"}});
    cs.stage_macroblock(FileSegment("Compared to the last puzzle, it's _much harder_."), 1);
    cs.render_microblock();

    // Looping animation scene - me and coworker
    LoopAnimationScene las({"coworker1", "coworker2", "coworker3", 
                            "give1", "give2", "give3",
                            "trying1", "trying2", "trying3",
                            "solved1", "solved2", "solved3",
                            "dizzy1", "dizzy2", "dizzy3"});
    las.state_manager.set({{"loop_length", "3"}});
    las.stage_macroblock(FileSegment("I showed it to a coworker and went home,"), 2);
    las.render_microblock();
    las.state_manager.set({{"loop_start", "3"}});
    las.render_microblock();

    // Looping animation scene - coworker struggling to solve
    las.state_manager.set({{"loop_start", "6"}});
    las.stage_macroblock(FileSegment("but he refused to leave until he solved it."), 1);
    las.render_microblock();

    // Looping animation scene - coworker dizzy, puzzle solved
    las.stage_macroblock(FileSegment("He finally got it... around 11:00 PM."), 2);
    las.state_manager.set({{"loop_start", "9"}});
    las.render_microblock();
    las.state_manager.set({{"loop_start", "12"}});
    las.render_microblock();
}

void part4() {
    CompositeScene cs;
    // Transition to subpuzzle containing only blocks b and d
    shared_ptr<KlotskiScene> ks_ptr = make_shared<KlotskiScene>(sun);
    ks_ptr = make_shared<KlotskiScene>(sun);
    cs.add_scene(ks_ptr, "ks", .5, -.5);
    cs.fade_all_subscenes(0);
    auto ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene(ks_bd_ptr, "ks_bd");
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    cs.remove_subscene("ks");

    // Animate big piece going under small piece.
    cs.stage_macroblock(FileSegment("He thought the hardest part was moving the box under the horizontal bar."), 6);
    ks_bd_ptr->stage_move({'b', 1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'d', -1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', 0, 3});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', -1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', 0, 6});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'d', 1, 0});
    cs.render_microblock();

    // Fade ks back in from ks_bd.
    cs.fade_all_subscenes(0);
    cs.add_scene_fade_in(ks_ptr, "ks", 0.5, 0.5);
    cs.stage_macroblock(FileSegment("I still haven't bothered solving it myself..."), 1);
    cs.render_microblock();

    // Start to grow a graph (a hundred nodes or so) in the background
    cs.stage_macroblock(FileSegment("I was more interested in seeing how it works under the hood."), 100);
    Graph bg_graph;
    bg_graph.add_to_stack(new KlotskiBoard(sun));
    auto bggs_ptr = make_shared<GraphScene>(&bg_graph);
    bggs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(bggs_ptr, "bggs");
    while(cs.microblocks_remaining()){
        bg_graph.expand_once();
        cs.render_microblock();
    }
    cs.stage_macroblock(FileSegment("What makes it so hard?"), 1);
    cs.state_manager.microblock_transition({{"ks.opacity", "0"}});
    cs.render_microblock();
    cs.stage_macroblock(FileSegment("Is getting the box under the bar actually the hardest part?"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileSegment("The structure defined by this puzzle- what is its _form_?"), 100);
    cs.fade_subscene("bggs", 0, false);
    while(cs.microblocks_remaining()){
        bg_graph.expand_once();
        cs.render_microblock();
    }
}

void showcase_graph(const KlotskiBoard& kb, const string& script) {
    CompositeScene cs;
    // Create a KlotskiScene for the given board and set its state
    auto ks_ptr = make_shared<KlotskiScene>(kb);
    ks_ptr->state_manager.set(board_width_height);
    cs.add_scene_fade_in(ks_ptr, "ks");
    cs.state_manager.set(board_position);

    // Create a graph starting from this board
    Graph g;
    g.add_to_stack(new KlotskiBoard(kb));
    auto gs_ptr = make_shared<GraphScene>(&g);
    gs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(gs_ptr, "gs");

    // Gradually expand the graph to reveal its structure
    int expansion_steps = get_graph_size(kb);
    cs.stage_macroblock(FileSegment(script), expansion_steps);
    while (cs.microblocks_remaining()) {
        g.expand_once();
        cs.render_microblock();
    }
    cs.stage_macroblock(SilenceSegment(4), 1);
    cs.render_microblock();
    cs.fade_all_subscenes(0);
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

void part5() {
    CompositeScene cs;

    // Add a klotski scene for manifold_1d
    auto ks1d = make_shared<KlotskiScene>(manifold_1d);
    ks1d->state_manager.set(board_width_height);
    cs.stage_macroblock(SilenceSegment(.8), 1);
    cs.add_scene_fade_in(ks1d, "ks1d");
    cs.state_manager.set({{"ks1d.x",".15"},{"ks1d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.render_microblock();

    Graph g1d;
    g1d.add_to_stack(new KlotskiBoard(manifold_1d));
    auto gs1d = make_shared<GraphScene>(&g1d);
    gs1d->state_manager.set(default_graph_state);
    cs.add_scene(gs1d, "gs1d", .6, .5);
    gs1d->next_hash = ks1d->copy_board().get_hash();

    cs.stage_macroblock(FileSegment("To help build some intuition, here are some contrived puzzles first."), get_graph_size(manifold_1d));
    while(cs.microblocks_remaining()) {
        g1d.expand_once();
        cs.render_microblock();
    }
    cs.stage_macroblock(FileSegment("To start off, with just a single long block, there is only one degree of freedom in movement."), 10);
    for(int i = 0; i < 10; i++){
        ks1d->stage_move({'a', 0, i<=4?1:-1});
        gs1d->next_hash = ks1d->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Fade out 1D and slide in 2D, then build 2D graph
    cs.stage_macroblock(FileSegment("With two,"), 1);
    cs.fade_all_subscenes(0);
    auto ks2d = make_shared<KlotskiScene>(manifold_2d);
    ks2d->state_manager.set(board_width_height);
    cs.add_scene_fade_in(ks2d, "ks2d");
    cs.state_manager.set({{"ks2d.x",".15"},{"ks2d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.render_microblock();
    cs.remove_subscene("ks1d");
    cs.remove_subscene("gs1d");

    cs.stage_macroblock(FileSegment("you get the cartesian product of both pieces, yielding a grid."), get_graph_size(manifold_2d));
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    auto gs2d = make_shared<GraphScene>(&g2d);
    gs2d->state_manager.set(default_graph_state);
    gs2d->next_hash = ks2d->copy_board().get_hash();
    cs.add_scene(gs2d, "gs2d", .6, .5);
    cs.render_microblock();
    while(cs.microblocks_remaining()) {
        g2d.expand_once();
        cs.render_microblock();
    }

    cs.stage_macroblock(FileSegment("Each block defines an axis of motion."), 20);
    for(int i = 0; i < 10; i++){
        ks2d->stage_move({'a', 0, i<=4?1:-1});
        gs2d->next_hash = ks2d->copy_staged_board().get_hash();
        cs.render_microblock();
    }
    for(int i = 0; i < 10; i++){
        ks2d->stage_move({'c', 0, i<=4?1:-1});
        gs2d->next_hash = ks2d->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Slide in 3D and leave a pause
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.state_manager.macroblock_transition({{"ks2d.x","1"},{"ks2d.y","-1"}});
    cs.state_manager.macroblock_transition({{"gs2d.x","1"},{"gs2d.y","-1"}});
    auto ks3d = make_shared<KlotskiScene>(manifold_3d);
    ks3d->state_manager.set(board_width_height);
    cs.add_scene(ks3d, "ks3d", -1, 1);
    Graph g3d;
    g3d.add_to_stack(new KlotskiBoard(manifold_3d));
    g3d.expand_completely();
    auto gs3d = make_shared<GraphScene>(&g3d);
    gs3d->state_manager.set(default_graph_state);
    cs.add_scene(gs3d, "gs3d", -1, 1);
    cs.state_manager.macroblock_transition({{"ks3d.x",".15"},{"ks3d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.macroblock_transition({{"gs3d.x",".6"},{"gs3d.y",".5"}});
    cs.render_microblock();
    gs2d->next_hash = 0;

    // Show it off a sec
    cs.stage_macroblock(FileSegment("Three blocks make for a 3d grid,"), 1);
    cs.render_microblock();

    // 4D hypercube
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.state_manager.macroblock_transition({{"ks3d.x","1"},{"ks3d.y","-1"}});
    cs.state_manager.macroblock_transition({{"gs3d.x","1"},{"gs3d.y","-1"}});
    auto ks4d = make_shared<KlotskiScene>(manifold_4d);
    ks4d->state_manager.set(board_width_height);
    cs.add_scene(ks4d, "ks4d", -1, 1);
    Graph g4d;
    g4d.add_to_stack(new KlotskiBoard(manifold_4d));
    g4d.expand_completely();
    auto gs4d = make_shared<GraphScene>(&g4d);
    gs4d->state_manager.set(default_graph_state);
    cs.add_scene(gs4d, "gs4d", -1, 1);
    cs.state_manager.macroblock_transition({{"ks4d.x",".15"},{"ks4d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.macroblock_transition({{"gs4d.x",".6"},{"gs4d.y",".5"}});
    cs.render_microblock();
    cs.remove_subscene("ks3d");
    cs.remove_subscene("gs3d");

    cs.stage_macroblock(FileSegment("and with 4 degrees of freedom, the graph naturally extends to a hypercube!"), 1);
    cs.render_microblock();

    // Bring back 2D without recreating it
    cs.stage_macroblock(FileSegment("But things get more fun when the pieces are capable of intersection."), 2);
    cs.state_manager.microblock_transition({{"ks4d.x","1"},{"ks4d.y","-1"}});
    cs.state_manager.microblock_transition({{"gs4d.x","1"},{"gs4d.y","-1"}});
    cs.state_manager.macroblock_transition({{"ks2d.x",".15"},{"ks2d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.macroblock_transition({{"gs2d.x",".6"},{"gs2d.y",".5"}});
    cs.render_microblock();
    cs.render_microblock();

    // Move 2D to left, show ring_big on right
    cs.stage_macroblock(FileSegment("If we take our two-block puzzle,"), 1);
    cs.state_manager.macroblock_transition({{"ks2d.x",".25"},{"ks2d.y",".25"}});
    cs.state_manager.macroblock_transition({{"gs2d.x",".25"},{"gs2d.y",".75"}});
    gs2d->state_manager.macroblock_transition({{"w",".5"},{"h",".5"}});
    ks2d->state_manager.macroblock_transition({{"w",".5"},{"h",".5"}});
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("and put the blocks opposing each other,"), 1);
    auto ksr7 = make_shared<KlotskiScene>(ring_7x7);
    ksr7->state_manager.macroblock_transition({{"w",".5"},{"h",".5"}});
    cs.add_scene_fade_in(ksr7, "ksr7");
    cs.state_manager.set({{"ksr7.x",".75"},{"ksr7.y",".25"}});
    Graph gr7;
    gr7.add_to_stack(new KlotskiBoard(ring_7x7));
    gr7.expand_completely();
    auto gsr = make_shared<GraphScene>(&gr7, 0.5,0.5);
    gsr->state_manager.set(default_graph_state);
    cs.add_scene_fade_in(gsr, "gsr", 0.75, 0.75);
    while(cs.microblocks_remaining()) {
        gr7.expand_once();
        cs.render_microblock();
    }

    // Overlap invalid region
    cs.stage_macroblock(FileSegment("a section of the 2d structure is no longer valid, representing a state of overlapping pieces."), 1);
    cs.state_manager.macroblock_transition({{"ks2d.x","-.25"}, {"gs2d.x","-.25"}});
    cs.state_manager.macroblock_transition({{"gsr.x",".6"}, {"gsr.y",".5"}});
    gsr->state_manager.macroblock_transition({{"w","1"}, {"h","1"}});
    cs.state_manager.macroblock_transition({{"ksr7.x",".15"}, {"ksr7.y",to_string(yval)}});
    cs.render_microblock();

    cs.stage_macroblock(SilenceSegment(1), 2);
    ksr7->stage_move({'a', 0, 1});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'b', 2, 0});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceSegment(2), 2);
    ksr7->stage_move({'a', 0, 3});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'a', 0, -3});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();

    // Triangle puzzle
    cs.fade_all_subscenes(0);
    cs.stage_macroblock(FileSegment("If we put the two blocks on the same lane,"), get_graph_size(triangle));
    auto kstri = make_shared<KlotskiScene>(triangle);
    kstri->state_manager.set(board_width_height);
    cs.add_scene_fade_in(kstri, "kstri");
    cs.state_manager.set({{"kstri.x",".15"},{"kstri.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    Graph grt;
    grt.add_to_stack(new KlotskiBoard(triangle));
    auto gst = make_shared<GraphScene>(&grt);
    gst->state_manager.set(default_graph_state);
    cs.add_scene_fade_in(gst, "gst", 0.6, 0.5);
    gst->next_hash = kstri->copy_board().get_hash();
    while(cs.microblocks_remaining()) {
        grt.expand_once();
        cs.render_microblock();
    }
    cs.stage_macroblock(FileSegment("we get this triangle shape."), 1);
    cs.render_microblock();

    // Move top then bottom
    cs.stage_macroblock(FileSegment("Wherever the top block is, that serves as a bound for the bottom block."), 35);
    for(int j = 0; j < 5; j++) {
        for(int i=0; i<5-j; ++i){
            kstri->stage_move({'c', 0, -1});
            gst->next_hash = kstri->copy_staged_board().get_hash();
            cs.render_microblock();
        }
        for(int i=0; i<5-j; ++i){
            kstri->stage_move({'c', 0, 1});
            gst->next_hash = kstri->copy_staged_board().get_hash();
            cs.render_microblock();
        }
        kstri->stage_move({'a', 0, 1});
        gst->next_hash = kstri->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Mirror-triangle graph
    gst->state_manager.microblock_transition({{"repel",".1"}});
    cs.stage_macroblock(SilenceSegment(.5), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceSegment(2), get_graph_size(triangle_inv));
    grt.add_to_stack(new KlotskiBoard(triangle_inv));
    gst->next_hash = kstri->copy_staged_board().get_hash();
    while(cs.microblocks_remaining()){
        grt.expand_once();
        cs.render_microblock();
    }

    // Illegal move & mirror highlight
    gst->state_manager.microblock_transition({{"repel","1"}});
    gst->next_hash = 0;
    cs.stage_macroblock(FileSegment("With a graph like this, we implicitly create an imaginary counterpart structure,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("corresponding to the valid states which are unreachable without one block passing through the other."), 6);
    gst->state_manager.set({{"physics_multiplier","1"}});
    for(int i = 0; i < 6; i++) {
        const std::string s = std::string(9*i, '.') + "....cx.......c........a........a...." + std::string((5 - i)*9, '.');
        std::string s_dot = s;
        std::replace(s_dot.begin(), s_dot.end(), 'x', '.');
        std::string s_c = s;
        std::replace(s_c.begin(), s_c.end(), 'x', 'c');
        KlotskiBoard a(9, 9, s_dot, true);
        KlotskiBoard b(9, 9, s_c, true);
        grt.add_directed_edge(a.get_hash(), b.get_hash());
        grt.add_directed_edge(b.get_hash(), a.get_hash());
        cs.render_microblock();
    }
    cs.stage_macroblock(SilenceSegment(1), 1);
    gst->next_hash = kstri->copy_board().get_hash();
    cs.render_microblock();
    cs.stage_macroblock(SilenceSegment(1), 1);
    kstri->stage_move({'c', 0, -7});
    gst->next_hash = kstri->copy_staged_board().get_hash();
    cs.render_microblock();

    // 3-intersecting blocks example
    cs.stage_macroblock(FileSegment("For example,"), 1);
    cs.fade_all_subscenes(0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    cs.stage_macroblock(FileSegment("3 intersecting pieces still form a cube, there's just some excavated areas."), get_graph_size(iblock));
    auto ks3rb = make_shared<KlotskiScene>(iblock);
    ks3rb->state_manager.set(board_width_height);
    cs.add_scene(ks3rb, "ks3rb");
    cs.state_manager.set({{"ks3rb.x",".15"},{"ks3rb.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    Graph g3rb;
    g3rb.add_to_stack(new KlotskiBoard(iblock));
    auto gs3rb = make_shared<GraphScene>(&g3rb);
    gs3rb->state_manager.set(default_graph_state);
    cs.add_scene(gs3rb, "gs3rb", 0.6, 0.5);
    while(cs.microblocks_remaining()){
        g3rb.expand_once();
        cs.render_microblock();
    }
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();

    // Fade out all
    cs.stage_macroblock(FileSegment("But as the piece number gets higher, the dimensionality of the graph has less and less to do with the number of pieces, and more to do with the number of unblocked pieces."), 1);
    cs.fade_all_subscenes(0);
    cs.render_microblock();
}

void part6() {
    CompositeScene cs;

    // apk puzzle and full expansion
    auto ks_apk = make_shared<KlotskiScene>(apk);
    ks_apk->state_manager.set(board_width_height);
    cs.add_scene_fade_in(ks_apk, "ks_apk");
    cs.state_manager.set({{"ks_apk.x",".15"},{"ks_apk.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    Graph g_apk;
    g_apk.add_to_stack(new KlotskiBoard(apk));
    auto gs_apk = make_shared<GraphScene>(&g_apk);
    gs_apk->state_manager.set(default_graph_state);
    //gs_apk->state_manager.microblock_transition({{"physics_multiplier","5"}});
    cs.add_scene(gs_apk, "gs_apk", 0.6, 0.5);
    gs_apk->state_manager.set({{"z_dilation", "1"}});

    cs.stage_macroblock(FileSegment("As an example, this puzzle has some cool behavior."), get_graph_size(apk));
    while(cs.microblocks_remaining()){
        g_apk.expand_once();
        cs.render_microblock();
    }

    cs.stage_macroblock(FileSegment("If I expand it out entirely,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("Notice that it has some overall superstructure,"), 1);
    cs.render_microblock();

    gs_apk->next_hash = ks_apk->copy_board().get_hash();

    gs_apk->state_manager.set({{"centering_strength","0"}});
    unordered_map<double,Node> nodes_copy = g_apk.nodes;
    list<double> to_remove;
    for(auto it = nodes_copy.begin(); it != nodes_copy.end(); ++it){
        double id_here = it->first;
        if(g_apk.dist(id_here, ks_apk->copy_board().get_hash()) < 20) continue;
        to_remove.push_back(id_here);
    }
    g_apk.clear_queue();

    cs.stage_macroblock(FileSegment("but also, if we zoom in,"), to_remove.size());
    gs_apk->state_manager.microblock_transition({{"z_dilation", ".98"}});
    for(double d : to_remove){
        g_apk.remove_node(d);
        cs.render_microblock();
    }

    gs_apk->state_manager.microblock_transition({{"centering_strength","0.1"}});
    cs.stage_macroblock(FileSegment("the local behavior is quite nicely patterned as well."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("It's a cute little local euclidean manifold with two degrees of freedom."), 1);
    cs.render_microblock();

    // show available actions on puzzle
    cs.stage_macroblock(FileSegment("On the puzzle, there are correspondingly only two available actions for each open hole."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("One axis characterized by moving the top hole,"), 4);
    ks_apk->stage_move({'c', 1, 0});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_apk->stage_move({'c', -1, 0});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_apk->stage_move({'a', -1, 0});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_apk->stage_move({'a', 1, 0});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();

    cs.stage_macroblock(FileSegment("One axis for the bottom."), 4);
    ks_apk->stage_move({'h', 1, 0});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_apk->stage_move({'h', -1, 0});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_apk->stage_move({'e', 0, 1});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_apk->stage_move({'e', 0, -1});
    gs_apk->next_hash = ks_apk->copy_staged_board().get_hash();
    cs.render_microblock();

    // swap to full_15_puzzle
    cs.stage_macroblock(SilenceSegment(1), 1);
    auto ks15 = make_shared<KlotskiScene>(full_15_puzzle);
    ks15->state_manager.set(board_width_height);
    ks15->state_manager.set({{"rainbow", "0"}});
    cs.add_scene(ks15, "ks15", 1.85, yval);
    cs.state_manager.microblock_transition({{"gs_apk.x","-.5"}, {"ks_apk.x","-.15"}, {"ks15.x",".85"}});
    cs.render_microblock();

    cs.remove_subscene("ks_apk");
    cs.remove_subscene("gs_apk");
    cs.stage_macroblock(FileSegment("In the extreme case, it's not the pieces moving, but rather the empty spaces which define the degrees of freedom."), get_graph_size(full_15_puzzle));
    Graph g_15;
    g_15.add_to_stack(new KlotskiBoard(full_15_puzzle));
    auto gs_15 = make_shared<GraphScene>(&g_15);
    gs_15->state_manager.set(default_graph_state);
    cs.add_scene(gs_15, "gs_15", 0.4, 0.5);
    while(cs.microblocks_remaining()){
        g_15.expand_once();
        cs.render_microblock();
    }

    // side‐by‐side: manifold_1d, rushhour_advanced, full_15_puzzle
    cs.stage_macroblock(FileSegment("So, somewhere in between a full board and an empty one, we get complex structures of tangled intersections between pieces."), 1);
    auto ks2d = make_shared<KlotskiScene>(manifold_2d, .3, .3);
    cs.add_scene_fade_in(ks2d, "ks2d");
    cs.state_manager.set({{"ks2d.x",".1666"},{"ks2d.y",to_string(yval)}});
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    auto gs2d = make_shared<GraphScene>(&g2d);
    gs2d->state_manager.set(default_graph_state);
    g2d.expand_completely();
    ks_apk->state_manager.set({{"w",".3"}, {"h",".5"}});
    ks2d->state_manager.set({{"w",".3"}, {"h",".5"}});
    ks15->state_manager.macroblock_transition({{"w",".3"}, {"h",".5"}});
    gs2d->state_manager.macroblock_transition({{"w",".3"}, {"h",".5"}});
    gs_apk->state_manager.macroblock_transition({{"w",".3"}, {"h",".5"}});
    gs_15->state_manager.macroblock_transition({{"w",".3"}, {"h",".5"}});
    cs.add_scene_fade_in(ks_apk, "ks_apk", .5, yval);
    cs.add_scene_fade_in(gs_apk, "gs_apk", .5, .75);
    cs.add_scene_fade_in(gs2d, "gs2d", .1666, .75);
    cs.state_manager.macroblock_transition({{"ks2d.x",".1666"}, {"ks_apk.x",".5"}, {"ks15.x",".8333"}, {"gs_15.x",".8333"}, {"gs_15.y",".75"}});
    cs.render_microblock();

    cs.stage_macroblock(SilenceSegment(.5), 1);
    cs.fade_all_subscenes(0);
    cs.render_microblock();
}

void part7() {
    CompositeScene cs;

    // intermediate graph overlay
    cs.stage_macroblock(FileSegment("This is the puzzle we started with."), 1/*get_graph_size(intermediate)*/);
    auto ks_int = make_shared<KlotskiScene>(intermediate);
    //ks_int->state_manager.set(board_width_height);
    cs.add_scene_fade_in(ks_int, "ks_int");
    cs.render_microblock();
    /*Graph g_int;
    g_int.add_to_stack(new KlotskiBoard(intermediate));
    auto gs_int = make_shared<GraphScene>(&g_int);
    gs_int->state_manager.set(default_graph_state);
    cs.add_scene(gs_int, "gs_int");
    while(cs.microblocks_remaining()) {
        g_int.expand_once();
        cs.render_microblock();
    }*/

    // grow five different graphs and overlay
    cs.stage_macroblock(SilenceSegment(.5), 1);
    vector<const KlotskiBoard*> boards = {&weird1, &euler766_easy, &beginner, &diamond};
    vector<string> names = {"w1","eul","beg","dia"};
    for(int i=0;i<boards.size();++i){
        Graph* g = new Graph();
        g->add_to_stack(new KlotskiBoard(*boards[i]));
        auto gs = make_shared<GraphScene>(g);
        gs->state_manager.set(default_graph_state);
        cs.add_scene_fade_in(gs, names[i]);
    }
    cs.render_microblock();

    // question mark
    cs.stage_macroblock(FileSegment("It has a very well-defined superstructure."), 1);
    cs.render_microblock();

    // pause scene
    cs.stage_macroblock(FileSegment("Take a moment to think through what it might be. You might be able to guess its form from the arrangement of the pieces!"), 1);
    shared_ptr<PauseScene> ps = make_shared<PauseScene>();
    cs.add_scene(ps, "ps");
    cs.render_microblock();
    cs.remove_subscene("ps");

/*
fff..c
..a..c
bba...
dda..e
.....e
hhh..e
*/

    perform_shortest_path(ks_int, KlotskiBoard(6, 6, "fff..c..a..cbba...dda..e.....ehhh..e", true), "...");
    KlotskiBoard bd_only(6, 6, "............bb....dd................", true);
    shared_ptr<KlotskiScene> bds = make_shared<KlotskiScene>(bd_only);
    cs.stage_macroblock(FileSegment("From this perspective, the puzzle is more or less symmetrical."), 1);
    cs.render_microblock();
    cs.fade_all_subscenes(.3);
    cs.add_scene_fade_in(bds, "bds");
    cs.stage_macroblock(FileSegment("The key is recognizing that these two pieces stay latched in one of two spots."), 1);
    cs.render_microblock();
    cs.fade_subscene("bds", 0);
    cs.fade_subscene("ks_int", 1);
    cs.add_scene(make_shared(*ks_int), "copy");
    cs.fade_subscene("copy", 0.5);
    cs.stage_macroblock(FileSegment("They can either be here,"), 1);
    cs.render_microblock();
    perform_shortest_path(ks_int, KlotskiBoard(6, 6, "fff..c..a..c..abb...adde.....ehhh..e", true), "or they can be here.");
    cs.stage_macroblock(FileSegment("This red vertical bar acts as a gate, permitting them to transition between the two states."), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileSegment("Furthermore, it can only permit one to transition at a time."), 1);
    cs.render_microblock();
}

void part8() {
    // Show the full klotski "sun" puzzle for the first time.
    CompositeScene cs_sun;
    auto ks_sun = make_shared<KlotskiScene>(sun);
    cs_sun.add_scene(ks_sun, "ks_sun");
    cs_sun.stage_macroblock(FileSegment("Alright, so what about Klotski?"), 5);
    // Demonstrate a few random moves on the sun puzzle.
    for (int i = 0; i < 5; i++) {
        ks_sun->stage_random_move();
        cs_sun.render_microblock();
    }
    cs_sun.fade_all_subscenes(0);
    cs_sun.stage_macroblock(SilenceSegment(1), 1);
    cs_sun.render_microblock();
}

void showcase_all_graphs(){
    for (const auto& kb_set : {gpt, other, suns, rushhours, geometry}) {
        for (const KlotskiBoard& kb : kb_set) {
            showcase_graph(kb, "aaaa");
        }
    }
}

void render_video() {
    //FOR_REAL = false;
    //PRINT_TO_TERMINAL = false;
    /*
    part1();
    part2();
    part3();
    part4();
    part5();
    part6();
    */
    part7();
    part8();
}
