#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/PauseScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Media/LoopAnimationScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"

int get_graph_size(const KlotskiBoard& kb){
    Graph g;
    g.add_to_stack(new KlotskiBoard(kb));
    g.expand();
    return g.size();
}

StateSet default_graph_state{
    {"q1", "1"},
    {"qi", "<t> .2 * cos"},
    {"qj", "<t> .314 * sin"},
    {"qk", "0"}, // Camera orientation quaternion
    {"decay",".93"},
    {"surfaces_opacity","0"}, // Whether we want to draw the board at every node
    {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
};
StateSet less_spinny{
    {"qi", "<t> .27 * cos .2 *"},
    {"qj", "<t> .13 * sin .2 *"},
    {"qk", "<t> .31 * sin .5 *"},
};
StateSet spinny{
    {"qi", "<t> .8 * cos"},
    {"qj", "<t> .6 * sin"},
    {"qk", "<t> .9 * sin"},
};
StateSet board_width_height{{"w",".3"},{"h",to_string(.3*VIDEO_WIDTH/VIDEO_HEIGHT)}};
StateSet board_position    {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}};
double yval = .15*VIDEO_WIDTH/VIDEO_HEIGHT;

void perform_shortest_path_with_graph(CompositeScene& cs, shared_ptr<GraphScene> gs_ptr, shared_ptr<KlotskiScene> ks_ptr, KlotskiBoard end, const Macroblock &msg) {
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    g.expand();
    auto path = g.shortest_path(ks_ptr->copy_board().get_hash(), end.get_hash()).first;
    cs.stage_macroblock(msg, path.size()-1);
    path.pop_front();
    while(cs.microblocks_remaining()){
        double next = *(path.begin());
        path.pop_front();
        Node node = g.nodes.at(next);
        KlotskiBoard* next_board = dynamic_cast<KlotskiBoard*>(node.data);
        ks_ptr->stage_move(ks_ptr->copy_board().move_required_to_reach(*next_board));
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }
};

void perform_shortest_path(CompositeScene& cs, shared_ptr<KlotskiScene> ks_ptr, KlotskiBoard end, const Macroblock &msg) {
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    g.expand();
    auto path = g.shortest_path(ks_ptr->copy_board().get_hash(), end.get_hash()).first;
    cs.stage_macroblock(msg, path.size()-1);
    path.pop_front();
    while(cs.microblocks_remaining()){
        double next = *(path.begin());
        path.pop_front();
        Node node = g.nodes.at(next);
        KlotskiBoard* next_board = dynamic_cast<KlotskiBoard*>(node.data);
        ks_ptr->stage_move(ks_ptr->copy_board().move_required_to_reach(*next_board));
        cs.render_microblock();
    }
};

void part1(){
    // TODO let's see multiple graphs expanded fully in the first 10 seconds, like the short
    CompositeScene cs;

    // Add KlotskiScene and GraphScene for Intermediate puzzle, begin making 50 random moves.
    auto ks_ptr = make_shared<KlotskiScene>(intermediate);
    ks_ptr->state_manager.set(board_width_height);
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    auto gs_ptr = std::make_shared<GraphScene>(&g, true);
    gs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(gs_ptr, "gs");
    cs.add_scene(ks_ptr, "ks");
    cs.state_manager.set(board_position);
    cs.stage_macroblock(FileBlock("You're looking at a random agent exploring the state-space graph of a slidy puzzle."), 100);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges();
        // Highlight the node of the board on the state-space graph
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Set ks.highlight_char to highlight the block which needs to get freed.
    auto ksb_ptr= make_shared<KlotskiScene>(KlotskiBoard(6, 6, "............bb......................", true));
    ksb_ptr->state_manager.set(board_width_height);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.add_scene_fade_in(MICRO, ksb_ptr, "ksb");
    cs.state_manager.set({{"ksb.x",".15"},{"ksb.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.fade_subscene(MICRO, "ks", 0);
    cs.state_manager.transition(MICRO, {{"ks.x",".5"},{"ks.y",".5"}});
    cs.state_manager.transition(MICRO, {{"ksb.x",".5"},{"ksb.y",".5"}});
    cs.fade_subscene(MICRO, "gs", 0);
    ks_ptr->state_manager.transition(MICRO, {{"w","1"},{"h","1"}});
    ksb_ptr->state_manager.transition(MICRO, {{"w","1"},{"h","1"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It wants to free this block from the hole on the right side."), 3);
    cs.render_microblock();
    ksb_ptr->highlight_char = 'b';
    cs.render_microblock();
    ksb_ptr->stage_move({'b', 12, 0});
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ksb", 0);
    cs.fade_subscene(MICRO, "ks", 1);
    ks_ptr->highlight_char = 'a';
    cs.stage_macroblock(FileBlock("However, it can't do that yet, since this piece is in the way..."), 1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "gs", 1);
    cs.state_manager.transition(MICRO, board_position);
    ks_ptr->state_manager.transition(MICRO, board_width_height);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    // Delete all nodes of the graph except for the current one. Turn on surface opacity, and turn off edge opacity.
    cs.stage_macroblock(SilenceBlock(3), g.size()-1);
    gs_ptr->state_manager.transition(MACRO, {{"surfaces_opacity",".5"},{"lines_opacity","0"}});
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
    gs_ptr->state_manager.transition(MICRO, {{"centering_strength","0.1"}});
    cs.stage_macroblock(FileBlock("We'll represent the current position of the puzzle as a node."), 1);
    cs.render_microblock();

    // Make one move and insert it on the graph.
    ks_ptr->highlight_char = 'c';
    gs_ptr->state_manager.transition(MICRO, {{"centering_strength","1"}});
    gs_ptr->state_manager.transition(MICRO, {{"physics_multiplier","1"}});
    cs.stage_macroblock(FileBlock("If we make one move on the puzzle,"), 5);
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
    g.add_missing_edges();
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.stage_macroblock(FileBlock("we arrive at a different node."), 1);
    cs.render_microblock();
    gs_ptr->state_manager.transition(MICRO, {{"physics_multiplier","1"}});

    cs.stage_macroblock(FileBlock("Since these positions are just one move apart,"), 1);
    cs.render_microblock();

    // Turn edge opacity on.
    gs_ptr->state_manager.transition(MACRO, {{"lines_opacity","1"}});
    gs_ptr->state_manager.transition(MICRO, {{"physics_multiplier","5"}});
    cs.stage_macroblock(FileBlock("We draw an edge connecting the two nodes."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 2);
    ks_ptr->stage_move({'c', 0, -1});
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_ptr->stage_move({'c', 0, 1});
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.render_microblock();

    // Make a few random moves.
    gs_ptr->state_manager.transition(MACRO, {{"surfaces_opacity","0"}});
    cs.stage_macroblock(FileBlock("Each node is connected to a few more, and drawing them, we construct this labyrinth of paths."), 8);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges();
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Unhighlight and fade out the KlotskiScene
    gs_ptr->next_hash = 0;
    cs.state_manager.transition(MACRO, {{"ks.opacity","-2"}});
    cs.state_manager.transition(MACRO, {{"gs.opacity","0"}});
    // Expand the graph by one node until it is halfway complete. Fade out everything from the CompositeScene and then delete scenes when faded out.
    cs.stage_macroblock(FileBlock("You might start to wonder- if we add all the nodes, what would the graph look like?"), get_graph_size(intermediate) * .7);
    while(cs.microblocks_remaining()) {
        g.expand(1);
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
        auto gs2d_ptr = make_shared<GraphScene>(&g2d, false);
        gs2d_ptr->state_manager.set(default_graph_state);
        cs.add_scene(gs2d_ptr, "gs2d");
        cs.stage_macroblock(FileBlock("Maybe it has some overarching structure..."), get_graph_size(manifold_2d));
        while(cs.microblocks_remaining()){
            g2d.expand(1);
            cs.render_microblock();
        }

        // Transition the 2D grid scene to the left (by setting its width to .5 and moving its x position to .25)
        StateSet grid_transition{{"w","0.5"},{"x","0.25"}};
        gs2d_ptr->state_manager.transition(MICRO, grid_transition);
        cs.state_manager.transition(MICRO, {{"gs2d.x",".25"}});
        cs.stage_macroblock(FileBlock("such as a two-dimensional grid,"), 1);
        cs.render_microblock();
    }

    Graph g3d;
    {
        // Create new GraphScene for manifold_3d on the right side of the screen and fade it in while expanding the graph completely 
        cs.stage_macroblock(FileBlock("or a 3d crystal lattice!"), get_graph_size(manifold_3d));
        g3d.add_to_stack(new KlotskiBoard(manifold_3d));
        auto gs3d_ptr = make_shared<GraphScene>(&g3d, false, .5, 1);
        gs3d_ptr->state_manager.set(default_graph_state);
        cs.add_scene(gs3d_ptr, "gs3d", 0.75, 0.5);
        while(cs.microblocks_remaining()){
            g3d.expand(1);
            cs.render_microblock();
        }

        // Fade out all scenes and then delete them
        cs.fade_all_subscenes(MICRO, 0);
        cs.stage_macroblock(SilenceBlock(1), 1);
        cs.render_microblock();
        cs.remove_all_subscenes();
    }

    {
        // Fade in and expand a GraphScene for intermediate again, but this time override "physics_multiplier" to be zero so the graph structure is indiscernable.
        cs.stage_macroblock(FileBlock("Maybe it's a dense mesh of interconnected nodes with no grand structure."), 100);
        Graph g_int;
        g_int.add_to_stack(new KlotskiBoard(intermediate));
        auto gs_int_ptr = make_shared<GraphScene>(&g_int, false);
        gs_int_ptr->state_manager.set(default_graph_state);
        gs_int_ptr->state_manager.set({{"attract","0"}, {"repel","0"}});
        //gs_int_ptr->state_manager.set({{"attract","-1"}, {"repel","-1"}});
        cs.add_scene_fade_in(MICRO, gs_int_ptr, "gs_int");
        while(cs.microblocks_remaining()){
            g_int.expand(1);
            cs.render_microblock();
        }

        // Fade out all scenes and then delete them
        cs.fade_all_subscenes(MICRO, 0);
        cs.stage_macroblock(SilenceBlock(1), 1);
        cs.render_microblock();
        cs.remove_all_subscenes();
    }
}

void part3() {
    // Start over by adding a KlotskiScene.
    shared_ptr<KlotskiScene> ks_ptr = make_shared<KlotskiScene>(sun);

    CompositeScene cs;
    cs.add_scene_fade_in(MICRO, ks_ptr, "ks");
    cs.stage_macroblock(SilenceBlock(.8), 1);
    cs.render_microblock();

    // Make moves according to the shortest path to the position given
    perform_shortest_path(cs, ks_ptr, KlotskiBoard(4, 5, "abbcabbc.gehj.ehddif", false), FileBlock("I fell down this rabbit hole when I was shown this puzzle."));

    // Make moves following the shortest path to the position given
    perform_shortest_path(cs, ks_ptr, KlotskiBoard(4, 5, "abbcabbcfidde.ghe.jh", false), FileBlock("It's called Klotski."));

    // Hotswap to a new KlotskiScene "ks2" with only the sun on it.
    auto ks2_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb.............", false));
    // Show piece 'b' getting moved 3 units downward and back.
    cs.add_scene(ks2_ptr, "ks2");
    cs.stage_macroblock(FileBlock("The goal is to get this big piece out of the bottom."), 2);
    cs.fade_subscene(MICRO, "ks", 0);
    cs.render_microblock();
    ks2_ptr->stage_move({'b', 0, 8});
    cs.render_microblock();

    // Swap back to ks. Transition "dots" state element to 1 and back to 0 to introduce the pins over two microblocks
    cs.stage_macroblock(FileBlock("It's slightly different since the intersections don't have pins,"), 3);
    cs.state_manager.transition(MICRO, {{"ks.opacity","1"},});
    cs.render_microblock();
    ks_ptr->state_manager.transition(MICRO, {{"dots", "1"}});
    cs.render_microblock();
    ks_ptr->state_manager.transition(MICRO, {{"dots", "0"}});
    cs.render_microblock();

    // Now that dots are back to 0, demonstrate a lateral move. (move piece 'e' right one space)
    ks_ptr->stage_move({'e', 1, 0});
    ks_ptr->stage_macroblock(FileBlock("so blocks are free to move laterally."), 1);
    ks_ptr->render_microblock();

    // Show the intermediate puzzle from before.
    cs.remove_all_subscenes();
    cs.add_scene(ks_ptr, "ks");
    cs.add_scene_fade_in(MICRO, make_shared<KlotskiScene>(intermediate, .5, 1), "ks_intermediate", .25, .5);
    cs.state_manager.transition(MICRO, {{"ks.x",".75"}});
    ks_ptr->state_manager.transition(MICRO, {{"w",".5"}});
    cs.stage_macroblock(FileBlock("Compared to the last puzzle, it's _much harder_."), 1);
    cs.render_microblock();

    // Looping animation scene - me and coworker
    LoopAnimationScene las({"coworker1", "coworker2", "coworker3", 
                            "give1", "give2", "give3",
                            "trying1", "trying2", "trying3",
                            "solved1", "solved2", "solved3",
                            "dizzy1", "dizzy2", "dizzy3"});
    las.state_manager.set({{"loop_length", "3"}});
    las.stage_macroblock(FileBlock("I showed it to a coworker and went home,"), 2);
    las.render_microblock();
    las.state_manager.set({{"loop_start", "3"}});
    las.render_microblock();

    // Looping animation scene - coworker struggling to solve
    las.state_manager.set({{"loop_start", "6"}});
    las.stage_macroblock(FileBlock("but he refused to leave until he solved it."), 1);
    las.render_microblock();

    // Looping animation scene - coworker dizzy, puzzle solved
    las.stage_macroblock(FileBlock("He finally got it... around 11:00 PM."), 2);
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
    cs.fade_all_subscenes(MICRO, 0);
    auto ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene(ks_bd_ptr, "ks_bd");
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_subscene("ks");

    // Animate big piece going under small piece.
    cs.stage_macroblock(FileBlock("He thought the hardest part was moving the box under the horizontal bar."), 6);
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
    cs.fade_all_subscenes(MICRO, 0);
    cs.add_scene_fade_in(MICRO, ks_ptr, "ks", 0.5, 0.5);
    cs.stage_macroblock(FileBlock("I still haven't bothered solving it myself..."), 1);
    cs.render_microblock();

    // Start to grow a graph (a hundred nodes or so) in the background
    cs.stage_macroblock(FileBlock("I was more interested in seeing how it works under the hood."), 100);
    Graph bg_graph;
    bg_graph.add_to_stack(new KlotskiBoard(sun));
    auto bggs_ptr = make_shared<GraphScene>(&bg_graph, false);
    bggs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(bggs_ptr, "bggs");
    while(cs.microblocks_remaining()){
        bg_graph.expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(FileBlock("What makes it so hard?"), 1);
    cs.state_manager.transition(MICRO, {{"ks.opacity", "0"}});
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("Is getting the box under the bar actually the hardest part?"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("The structure defined by this puzzle- what is its _form_?"), 100);
    cs.fade_subscene(MACRO, "bggs", 0);
    while(cs.microblocks_remaining()){
        bg_graph.expand(1);
        cs.render_microblock();
    }
}

void showcase_graph(const KlotskiBoard& kb, const Macroblock& mb) {
    CompositeScene cs;
    // Create a KlotskiScene for the given board and set its state
    auto ks_ptr = make_shared<KlotskiScene>(kb);
    ks_ptr->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks_ptr, "ks");
    cs.state_manager.set(board_position);

    // Create a graph starting from this board
    Graph g;
    g.add_to_stack(new KlotskiBoard(kb));
    auto gs_ptr = make_shared<GraphScene>(&g, false);
    gs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(gs_ptr, "gs");

    shared_ptr<LatexScene> ls = make_shared<LatexScene>("Node Count: " + get_graph_size(kb), 1, .3, .3);
    cs.add_scene(ls, "ls", .15, .15);

    // Gradually expand the graph to reveal its structure
    int expansion_steps = get_graph_size(kb);
    cs.stage_macroblock(mb, expansion_steps);
    while (cs.microblocks_remaining()) {
        g.expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(SilenceBlock(4), 1);
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

void part5() {
    CompositeScene cs;

    // Add a klotski scene for manifold_1d
    auto ks1d = make_shared<KlotskiScene>(manifold_1d);
    ks1d->state_manager.set(board_width_height);
    cs.stage_macroblock(SilenceBlock(.8), 1);
    cs.add_scene_fade_in(MICRO, ks1d, "ks1d");
    cs.state_manager.set({{"ks1d.x",".15"},{"ks1d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.render_microblock();

    Graph g1d;
    g1d.add_to_stack(new KlotskiBoard(manifold_1d));
    auto gs1d = make_shared<GraphScene>(&g1d, false);
    gs1d->state_manager.set(default_graph_state);
    cs.add_scene(gs1d, "gs1d", .6, .5);
    gs1d->next_hash = ks1d->copy_board().get_hash();

    cs.stage_macroblock(FileBlock("To help build some intuition, here are some contrived puzzles first."), get_graph_size(manifold_1d));
    while(cs.microblocks_remaining()) {
        g1d.expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(FileBlock("To start off, with just a single long block, there's only one degree of freedom in movement."), 10);
    for(int i = 0; i < 10; i++){
        ks1d->stage_move({'a', 0, i<=4?1:-1});
        gs1d->next_hash = ks1d->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Fade out 1D and slide in 2D, then build 2D graph
    cs.stage_macroblock(FileBlock("With two,"), 1);
    cs.fade_all_subscenes(MICRO, 0);
    auto ks2d = make_shared<KlotskiScene>(manifold_2d);
    ks2d->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks2d, "ks2d");
    cs.state_manager.set({{"ks2d.x",".15"},{"ks2d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.render_microblock();
    cs.remove_subscene("ks1d");
    cs.remove_subscene("gs1d");

    cs.stage_macroblock(FileBlock("you get the cartesian product of both pieces, yielding a grid."), get_graph_size(manifold_2d));
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    auto gs2d = make_shared<GraphScene>(&g2d, false);
    gs2d->state_manager.set(default_graph_state);
    gs2d->next_hash = ks2d->copy_board().get_hash();
    cs.add_scene(gs2d, "gs2d", .6, .5);
    cs.render_microblock();
    while(cs.microblocks_remaining()) {
        g2d.expand(1);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("Each block defines a different axis of motion."), 20);
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
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.state_manager.transition(MACRO, {{"ks2d.x","1"},{"ks2d.y","-1"}});
    cs.state_manager.transition(MACRO, {{"gs2d.x","1"},{"gs2d.y","-1"}});
    auto ks3d = make_shared<KlotskiScene>(manifold_3d);
    ks3d->state_manager.set(board_width_height);
    cs.add_scene(ks3d, "ks3d", -1, 1);
    Graph g3d;
    g3d.add_to_stack(new KlotskiBoard(manifold_3d));
    g3d.expand();
    auto gs3d = make_shared<GraphScene>(&g3d, false);
    gs3d->state_manager.set(default_graph_state);
    cs.add_scene(gs3d, "gs3d", -1, 1);
    cs.state_manager.transition(MACRO, {{"ks3d.x",".15"},{"ks3d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MACRO, {{"gs3d.x",".6"},{"gs3d.y",".5"}});
    cs.render_microblock();
    gs2d->next_hash = 0;

    // Show it off a sec
    cs.stage_macroblock(FileBlock("Three blocks make for a 3d grid,"), 1);
    cs.render_microblock();

    // 4D hypercube
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.state_manager.transition(MACRO, {{"ks3d.x","1"},{"ks3d.y","-1"}});
    cs.state_manager.transition(MACRO, {{"gs3d.x","1"},{"gs3d.y","-1"}});
    auto ks4d = make_shared<KlotskiScene>(manifold_4d);
    ks4d->state_manager.set(board_width_height);
    cs.add_scene(ks4d, "ks4d", -1, 1);
    Graph g4d;
    g4d.add_to_stack(new KlotskiBoard(manifold_4d));
    g4d.expand();
    auto gs4d = make_shared<GraphScene>(&g4d, false);
    gs4d->state_manager.set(default_graph_state);
    cs.add_scene(gs4d, "gs4d", -1, 1);
    cs.state_manager.transition(MACRO, {{"ks4d.x",".15"},{"ks4d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MACRO, {{"gs4d.x",".6"},{"gs4d.y",".5"}});
    cs.render_microblock();
    cs.remove_subscene("ks3d");
    cs.remove_subscene("gs3d");

    cs.stage_macroblock(FileBlock("and with 4 degrees of freedom, the graph naturally extends to a hypercube!"), 1);
    cs.render_microblock();

    // Bring back 2D without recreating it
    cs.stage_macroblock(FileBlock("But things get more fun when the pieces are capable of intersection."), 2);
    cs.state_manager.transition(MICRO, {{"ks4d.x","1"},{"ks4d.y","-1"}});
    cs.state_manager.transition(MICRO, {{"gs4d.x","1"},{"gs4d.y","-1"}});
    cs.state_manager.transition(MACRO, {{"ks2d.x",".15"},{"ks2d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MACRO, {{"gs2d.x",".6"},{"gs2d.y",".5"}});
    cs.render_microblock();
    cs.render_microblock();

    // Move 2D to left, show ring_big on right
    cs.stage_macroblock(FileBlock("If we take our two-block puzzle,"), 1);
    cs.state_manager.transition(MACRO, {{"ks2d.x",".25"},{"ks2d.y",".25"}});
    cs.state_manager.transition(MACRO, {{"gs2d.x",".25"},{"gs2d.y",".75"}});
    gs2d->state_manager.transition(MACRO, {{"w",".5"},{"h",".5"}});
    ks2d->state_manager.transition(MACRO, {{"w",".5"},{"h",".5"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and put the blocks opposing each other,"), 1);
    auto ksr7 = make_shared<KlotskiScene>(ring_7x7);
    ksr7->state_manager.transition(MACRO, {{"w",".5"},{"h",".5"}});
    cs.add_scene_fade_in(MICRO, ksr7, "ksr7");
    cs.state_manager.set({{"ksr7.x",".75"},{"ksr7.y",".25"}});
    Graph gr7;
    gr7.add_to_stack(new KlotskiBoard(ring_7x7));
    gr7.expand();
    auto gsr = make_shared<GraphScene>(&gr7, false, 0.5,0.5);
    gsr->state_manager.set(default_graph_state);
    cs.add_scene_fade_in(MICRO, gsr, "gsr", 0.75, 0.75);
    while(cs.microblocks_remaining()) {
        gr7.expand(1);
        cs.render_microblock();
    }

    // Overlap invalid region
    cs.stage_macroblock(FileBlock("a section of the 2d structure is no longer valid, representing a state of overlapping pieces."), 1);
    cs.state_manager.transition(MACRO, {{"ks2d.x","-.25"}, {"gs2d.x","-.25"}});
    cs.state_manager.transition(MACRO, {{"gsr.x",".6"}, {"gsr.y",".5"}});
    gsr->state_manager.transition(MACRO, {{"w","1"}, {"h","1"}});
    cs.state_manager.transition(MACRO, {{"ksr7.x",".15"}, {"ksr7.y",to_string(yval)}});
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 3);
    ksr7->stage_move({'a', 0, 1});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'b', 1, 0});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'b', 1, 0});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 2);
    ksr7->stage_move({'a', 0, 3});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'a', 0, -3});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();

;
    perform_shortest_path_with_graph(cs, gsr, ksr7, KlotskiBoard(7, 7, "...a......a..........bb..........................", true ), SilenceBlock(.5));
    perform_shortest_path_with_graph(cs, gsr, ksr7, KlotskiBoard(7, 7, ".....................bb...............a......a...", true ), SilenceBlock(1));
    perform_shortest_path_with_graph(cs, gsr, ksr7, KlotskiBoard(7, 7, "..........................bb..........a......a...", true ), SilenceBlock(1));
    perform_shortest_path_with_graph(cs, gsr, ksr7, KlotskiBoard(7, 7, "...a......a...............bb.....................", true ), SilenceBlock(1));
    perform_shortest_path_with_graph(cs, gsr, ksr7, KlotskiBoard(7, 7, "...a......a..........bb..........................", true ), SilenceBlock(1));

    // Triangle puzzle
    cs.fade_all_subscenes(MICRO, 0);
    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    //TODO cut out this section??
    cs.stage_macroblock(FileBlock("If we put the two blocks on the same lane,"), get_graph_size(triangle));
    auto kstri = make_shared<KlotskiScene>(triangle);
    kstri->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, kstri, "kstri");
    cs.state_manager.set({{"kstri.x",".15"},{"kstri.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    Graph grt;
    grt.add_to_stack(new KlotskiBoard(triangle));
    auto gst = make_shared<GraphScene>(&grt, false);
    gst->state_manager.set(default_graph_state);
    cs.add_scene_fade_in(MICRO, gst, "gst", 0.6, 0.5);
    gst->next_hash = kstri->copy_board().get_hash();
    while(cs.microblocks_remaining()) {
        grt.expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(FileBlock("we get this triangle shape."), 1);
    cs.render_microblock();

    // Move top then bottom
    cs.stage_macroblock(FileBlock("Wherever the top block is, that serves as a bound for the bottom block."), 35);
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
    gst->state_manager.transition(MICRO, {{"repel",".1"}});
    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceBlock(2), get_graph_size(triangle_inv));
    grt.add_to_stack(new KlotskiBoard(triangle_inv));
    gst->next_hash = kstri->copy_staged_board().get_hash();
    while(cs.microblocks_remaining()){
        grt.expand(1);
        cs.render_microblock();
    }

    // Illegal move & mirror highlight
    gst->state_manager.transition(MICRO, {{"repel","1"}});
    cs.stage_macroblock(FileBlock("With a graph like this, we implicitly create an imaginary counterpart structure,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("corresponding to the valid states which are unreachable without one block passing through the other."), 1);
    gst->state_manager.transition(MACRO, {{"physics_multiplier","1"}});
    /*for(int i = 0; i < 6; i++) {
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
    }*/
    cs.render_microblock();
    cs.stage_macroblock(SilenceBlock(1), 1);
    kstri->stage_move({'c', 0, -7});
    KlotskiBoard lie(9, 9, "....cc.......c...................................a........a......................", true );
    gst->next_hash = lie.get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    // 3-intersecting blocks example
    auto ks3rb = make_shared<KlotskiScene>(iblock);
    ks3rb->state_manager.set(board_width_height);
    cs.add_scene(ks3rb, "ks3rb");
    cs.state_manager.set({{"ks3rb.x",".15"},{"ks3rb.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.stage_macroblock(FileBlock("For example,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("3 intersecting pieces still form a cube, there's just some excavated areas."), get_graph_size(iblock));
    Graph g3rb;
    g3rb.add_to_stack(new KlotskiBoard(iblock));
    auto gs3rb = make_shared<GraphScene>(&g3rb, false);
    gs3rb->state_manager.set(default_graph_state);
    cs.add_scene(gs3rb, "gs3rb", 0.6, 0.5);
    while(cs.microblocks_remaining()){
        g3rb.expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    // Fade out all
    cs.stage_macroblock(FileBlock("But as the piece number gets higher, the dimensionality of the graph has less and less to do with the number of pieces, and more to do with the number of unblocked pieces."), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
}

void part6() {
    CompositeScene cs;

    // apk puzzle and full expansion
    auto ks_apk = make_shared<KlotskiScene>(apk);
    ks_apk->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks_apk, "ks_apk");
    cs.state_manager.set({{"ks_apk.x",".15"},{"ks_apk.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    Graph g_apk;
    g_apk.add_to_stack(new KlotskiBoard(apk));
    auto gs_apk = make_shared<GraphScene>(&g_apk, false);
    gs_apk->state_manager.set(default_graph_state);
    //gs_apk->state_manager.transition(MICRO, {{"physics_multiplier","5"}});
    cs.add_scene(gs_apk, "gs_apk", 0.6, 0.5);

    cs.stage_macroblock(FileBlock("As an example, this puzzle has some cool behavior."), get_graph_size(apk));
    while(cs.microblocks_remaining()){
        g_apk.expand(1);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("If I expand it out entirely,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Notice that it has some overall superstructure,"), 1);
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

    cs.stage_macroblock(FileBlock("but also, if we zoom in on a small portion of the graph,"), to_remove.size());
    // TODO zoom in more peacefully
    gs_apk->state_manager.transition(MICRO, {{"dimensions", "2.98"}});
    for(double d : to_remove){
        g_apk.remove_node(d);
        cs.render_microblock();
    }

    gs_apk->state_manager.transition(MICRO, {{"centering_strength","0.1"}});
    cs.stage_macroblock(FileBlock("the local behavior is quite nicely patterned as well."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It's a cute little local euclidean manifold with two dimensions- two degrees of freedom."), 1);
    cs.render_microblock();

    // TODO show available actions on puzzle or empty spaces
    // TODO show holes
    cs.stage_macroblock(FileBlock("That's because on the puzzle, there are two available holes to permit movement."), 1);
    cs.render_microblock();

    gs_apk->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("One axis characterized by moving the top hole,"), 4);
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

    cs.stage_macroblock(FileBlock("One axis for the bottom."), 4);
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
    cs.stage_macroblock(SilenceBlock(1), 1);
    auto ks15 = make_shared<KlotskiScene>(full_15_puzzle);
    ks15->state_manager.set(board_width_height);
    ks15->state_manager.set({{"rainbow", "0"}});
    cs.add_scene(ks15, "ks15", 1.85, yval);
    cs.state_manager.transition(MICRO, {{"gs_apk.x","-.5"}, {"ks_apk.x","-.15"}, {"ks15.x",".85"}});
    cs.render_microblock();

    cs.remove_subscene("ks_apk");
    cs.remove_subscene("gs_apk");
    cs.stage_macroblock(FileBlock("In the extreme case, it's not the pieces moving, but rather the empty spaces which define the degrees of freedom."), get_graph_size(full_15_puzzle));
    Graph g_15;
    g_15.add_to_stack(new KlotskiBoard(full_15_puzzle));
    auto gs_15 = make_shared<GraphScene>(&g_15, false);
    gs_15->state_manager.set(default_graph_state);
    cs.add_scene(gs_15, "gs_15", 0.4, 0.5);
    while(cs.microblocks_remaining()){
        g_15.expand(1);
        cs.render_microblock();
    }

    // side‐by‐side: manifold_1d, rushhour_advanced, full_15_puzzle
    cs.stage_macroblock(FileBlock("So, somewhere in between a full board and an empty one, we get complex structures of tangled intersections between pieces."), 1);
    auto ks2d = make_shared<KlotskiScene>(manifold_2d, .3, .3);
    cs.add_scene_fade_in(MICRO, ks2d, "ks2d");
    cs.state_manager.set({{"ks2d.x",".1666"},{"ks2d.y",to_string(yval)}});
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    auto gs2d = make_shared<GraphScene>(&g2d, false);
    gs2d->state_manager.set(default_graph_state);
    g2d.expand();
    ks_apk->state_manager.set({{"w",".3"}, {"h",".5"}});
    ks2d->state_manager.set({{"w",".3"}, {"h",".5"}});
    ks15->state_manager.transition(MACRO, {{"w",".3"}, {"h",".5"}});
    gs2d->state_manager.transition(MACRO, {{"w",".3"}, {"h",".5"}});
    gs_apk->state_manager.transition(MACRO, {{"w",".3"}, {"h",".5"}});
    gs_15->state_manager.transition(MACRO, {{"w",".3"}, {"h",".5"}});
    cs.add_scene_fade_in(MICRO, ks_apk, "ks_apk", .5, yval);
    cs.add_scene_fade_in(MICRO, gs_apk, "gs_apk", .5, .75);
    cs.add_scene_fade_in(MICRO, gs2d, "gs2d", .1666, .75);
    cs.state_manager.transition(MACRO, {{"ks2d.x",".1666"}, {"ks_apk.x",".5"}, {"ks15.x",".8333"}, {"gs_15.x",".8333"}, {"gs_15.y",".75"}});
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
}

void part7() {
    CompositeScene cs;

    // intermediate graph overlay
    cs.stage_macroblock(FileBlock("Here's the puzzle we started with."), 1/*get_graph_size(intermediate)*/);
    auto ks_int = make_shared<KlotskiScene>(intermediate);
    //ks_int->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks_int, "ks_int");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It has a very well-defined superstructure."), 1);
    cs.render_microblock();

    // pause scene
    cs.stage_macroblock(FileBlock("Take a moment to think through what it might look like. You might be able to guess its form from the arrangement of the pieces!"), 1);
    vector<const KlotskiBoard*> boards = {&weird1, &euler766_easy, &beginner, &diamond};
    vector<string> names = {"w1","eul","beg","dia"};
    for(int i=0;i<boards.size();++i){
        Graph* g = new Graph();
        g->add_to_stack(new KlotskiBoard(*boards[i]));
        g->expand();
        auto gs = make_shared<GraphScene>(g, false, .65, .65);
        gs->state_manager.set(default_graph_state);
        bool top = (i&1)==1;
        bool left = (i&2)==2;
        gs->state_manager.set({
            {"qi",string(top ?"":"-") + "<t> .2 * cos"},
            {"qj",string(left?"":"-") + "<t> .314 * sin"}
        });
        cs.add_scene_fade_in(MICRO, gs, names[i], top?.25:.75, left?.25:.75, .5);
    }
    shared_ptr<PauseScene> ps = make_shared<PauseScene>();
    cs.add_scene(ps, "ps");
    cs.render_microblock();
    cs.remove_subscene("ps");

    cs.stage_macroblock(SilenceBlock(.7), 1);
    cs.render_microblock();

    perform_shortest_path(cs, ks_int, KlotskiBoard(6, 6, "..fffc..a..cbba...dda..e.....e..hhhe", true), SilenceBlock(1));

    cs.stage_macroblock(SilenceBlock(.7), 1);
    cs.fade_all_subscenes_except(MICRO, "ks_int", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("ks_int");

/*
fff..c
..a..c
bba...
dda..e
.....e
hhh..e
*/

    KlotskiBoard bd_only(6, 6, "............bb....dd................", true);
    shared_ptr<KlotskiScene> bds = make_shared<KlotskiScene>(bd_only);
    ThreeDimensionScene tds;
    cs.remove_subscene("ks_int");
    tds.add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, .5, 0), "ks_int"), ks_int);
    tds.stage_macroblock(FileBlock("From this perspective, the puzzle is more or less symmetrical."), 5);
    tds.render_microblock();
    tds.render_microblock();
    tds.state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "1"}});
    tds.render_microblock();
    tds.render_microblock();
    tds.state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}});
    tds.render_microblock();

    tds.remove_surface("ks_int");
    cs.add_scene(ks_int, "ks_int");
    cs.fade_all_subscenes(MICRO, 0);
    cs.add_scene_fade_in(MICRO, bds, "bds");
    cs.stage_macroblock(FileBlock("The key is recognizing that these two pieces stay latched in one of two spots."), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.fade_subscene(MICRO, "bds", 0);
    cs.fade_subscene(MICRO, "ks_int", 1);
    shared_ptr<KlotskiScene> copy = make_shared<KlotskiScene>(KlotskiBoard(6,6,"..fffc..a..cbba...dda..e.....e..hhhe",true));
    cs.add_scene_fade_in(MICRO, copy, "copy", 0.5, 0.5, 0.25);
    cs.stage_macroblock(FileBlock("They can either be to the left of the vertical red bar,"), 1);
    cs.render_microblock();

    perform_shortest_path(cs, ks_int, KlotskiBoard(6, 6, "..fffc..a..c..abb...adde.....e..hhhe", true), FileBlock("or they can be to the right of it."));

    cs.stage_macroblock(SilenceBlock(.6), 1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "copy", 0);
    cs.stage_macroblock(FileBlock("This red bar acts as a gate, permitting them to transition between the two states."), 1);
    ks_int->highlight_char = 'a';
    cs.render_microblock();

    cs.remove_subscene("copy");
    ks_int->highlight_char = '.';
    cs.stage_macroblock(SilenceBlock(.7), 1);
    cs.render_microblock();

/*
..afff
..a..c
..abbc
...dde
.....e
hhh..e
*/

    perform_shortest_path(cs, ks_int, KlotskiBoard(6, 6, "..afff..a..c..abbc...dde.....e..hhhe", true), FileBlock("Only one can transition at a time."));

    cs.stage_macroblock(FileBlock("When the red bar is up, the orange block can transition."), 4);
    ks_int->stage_move({'d', -3, 0});
    cs.render_microblock();
    ks_int->stage_move({'d', 3, 0});
    cs.render_microblock();
    ks_int->stage_move({'d', -3, 0});
    cs.render_microblock();
    ks_int->stage_move({'d', 3, 0});
    cs.render_microblock();

/*
..fffc
.....c
...bbe
..adde
..a..e
..ahhh
*/

    perform_shortest_path(cs, ks_int, KlotskiBoard(6, 6, "..fffc.....c...bbe..adde..a..e..ahhh", true), SilenceBlock(1.5));

    cs.stage_macroblock(FileBlock("But when it's down, the green block can transition."), 4);
    ks_int->stage_move({'b', -3, 0});
    cs.render_microblock();
    ks_int->stage_move({'b', 3, 0});
    cs.render_microblock();
    ks_int->stage_move({'b', -3, 0});
    cs.render_microblock();
    ks_int->stage_move({'b', 3, 0});
    cs.render_microblock();

    cs.state_manager.transition(MACRO, {{"ks_int.x",".15"},{"ks_int.y",to_string(yval)}});
    ks_int->state_manager.transition(MACRO, board_width_height);
    cs.stage_macroblock(FileBlock("Let's build the graph."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), get_graph_size(intermediate));
    Graph g_int;
    g_int.add_to_stack(new KlotskiBoard(intermediate));
    auto gs_int = make_shared<GraphScene>(&g_int, false);
    gs_int->state_manager.set(default_graph_state);
    gs_int->state_manager.set(less_spinny);
    cs.add_scene(gs_int, "gs_int", .6, .5);
    while(cs.microblocks_remaining()) {
        g_int.expand(1);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("It's a square connecting 4 corners."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can color the nodes on the graph in correspondence with the position of the puzzle."), 1);
    //TODO before final render, turn GPU alpha back on
    gs_int->state_manager.transition(MACRO, {{"lines_opacity","0"}, {"points_radius_multiplier","2"}});
    for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0xff888888;
        if('b' == n.data->representation[16]) n.color |= 0xffff0000;
        if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
        if('d' == n.data->representation[22]) n.color |= 0xff00ff00;
        if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
    }
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Nodes in this half have the green block right of the bar,"), 1);
    for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x22888888;
        if('b' == n.data->representation[16]) n.color |= 0xffff0000;
        if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
        if('d' == n.data->representation[22]) n.color |= 0x0000ff00;
        if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
    }
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and nodes in this half have the orange block right of the bar."), 1);
    for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x22888888;
        if('b' == n.data->representation[16]) n.color |= 0x00ff0000;
        if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
        if('d' == n.data->representation[22]) n.color |= 0xff00ff00;
        if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
    }
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's take a tour around this graph."), 1);
    for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0xff888888;
        if('b' == n.data->representation[16]) n.color |= 0xffff0000;
        if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
        if('d' == n.data->representation[22]) n.color |= 0xff00ff00;
        if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
    }
    gs_int->state_manager.transition(MACRO, {{"lines_opacity",".5"}, {"points_radius_multiplier","1.5"}});
    gs_int->next_hash = ks_int->copy_staged_board().get_hash();
    cs.render_microblock();

    perform_shortest_path_with_graph(cs, gs_int, ks_int, intermediate, SilenceBlock(5));
    KlotskiBoard swaparound(6, 6, "..afff..a..c..abbcdd...e.....e..hhhe", true );
    perform_shortest_path_with_graph(cs, gs_int, ks_int, swaparound, SilenceBlock(5));
    KlotskiBoard swaparound2(6, 6, "..afff..a..cbba..c...dde.....e..hhhe", true );
    perform_shortest_path_with_graph(cs, gs_int, ks_int, swaparound2, SilenceBlock(5));

    cs.stage_macroblock(FileBlock("Now, from here, we can highlight all the solutions-"), 1);
    for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x22888888;
        if('b' == n.data->representation[17]) n.color |= 0xffff0000;
        if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
        if('d' == n.data->representation[22]) n.color |= 0x0000ff00;
        if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
    }
    cs.render_microblock();

/*
..fffc
.....c
..a.bb
..adde
..a..e
..hhhe
*/

    KlotskiBoard intermediate_solution(6, 6, "..fffc.....c..a.bb..adde..a..e..hhhe", true);

    gs_int->state_manager.transition(MACRO, {{"lines_opacity",".2"}});
    perform_shortest_path_with_graph(cs, gs_int, ks_int, intermediate_solution, SilenceBlock(5));

    cs.stage_macroblock(FileBlock("Unsurprisingly, they all live in the area with the green block right of the bar."), 1);
    for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x22888888;
        if('b' == n.data->representation[16]) n.color |= 0xffff0000;
        if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
        if('d' == n.data->representation[22]) n.color |= 0x0000ff00;
        if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
    }
    gs_int->next_hash = ks_int->copy_staged_board().get_hash();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Can you figure out why the graph has a higher dimensionality near the corners?"), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.8), 1);
    cs.render_microblock();
}

void part8() {
    //FOR_REAL = false;
    CompositeScene cs;
    auto ks = make_shared<KlotskiScene>(sun);
    cs.add_scene(ks, "ks", .5, 1.5);
    cs.slide_subscene(MACRO, "ks", 0, -1);
    cs.stage_macroblock(FileBlock("Alright, so what about Klotski?"), 1/*num_moves TODO*/);
    cs.render_microblock();

    ks->state_manager.transition(MICRO, board_width_height);
    cs .state_manager.transition(MICRO, {{"ks.x",".85"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    Graph g;
    g.add_to_stack(new KlotskiBoard(sun));
    auto gs = std::make_shared<GraphScene>(&g, false);
    gs->state_manager.set(default_graph_state);
    gs->state_manager.set({{"mirror_force", ".1"}, {"physics_multiplier", "40"}, {"points_opacity", "0"}, {"dimensions", "3.98"}});
    cs.add_scene(gs, "gs");

    int x = get_graph_size(sun);
    float i = 1;
    while (g.size() < x) {
        int num_nodes_to_add = i*i;
        g.expand(num_nodes_to_add);
        cs.stage_macroblock(SilenceBlock(.1), 1); // TODO change to 0.033333 before rendering, and third the num_nodes_to_add.
        cs.render_microblock();
        i+=.2;
        cout << g.size() << endl;
    }

    gs->state_manager.transition(MICRO, {{"physics_multiplier", "0"}, {"dimensions", "3.7"}});
    cs.stage_macroblock(SilenceBlock(15), 1);
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("That's 25,955 nodes."), 1);
    cs.render_microblock();

    // TODO show a symmetrical board pair!
    cs.stage_macroblock(FileBlock("The puzzle is symmetrical, so the graph is too."), 1);
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", ".5"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("Turning the graph 90 degrees, we see two primary clusters of nodes."), 1);
    cs.render_microblock();

    gs->next_hash = ks->copy_staged_board().get_hash();
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"d", ".2"}, {"points_opacity", "1"}, {"points_radius_multiplier","1.5"}});
    gs->state_manager.transition(MACRO, less_spinny);
    cs.stage_macroblock(FileBlock("Here's the starting position."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"lines_opacity", ".2"}});
    gs->state_manager.transition(MACRO, {{"d", "1.5"}});
    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", ".5"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("Now, let's look at all the solutions- the nodes with the square at the bottom."), 1);
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x00000000;
        if('b' == n.data->representation[13] && 'b' == n.data->representation[18]) n.color |= 0xff00ffff;
    }
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("All the solution nodes are on the opposite half of the graph as the starting position."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    gs->state_manager.transition(MACRO, {{"d", ".4"}});
    cs.render_microblock();

    const int num_random_moves = 40;
    cs.stage_macroblock(FileBlock("By making random moves from the starting position,"), num_random_moves);
    for (int i = 0; i < num_random_moves; i++) {
        ks->stage_random_move();
        gs->next_hash = ks->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    gs->state_manager.transition(MACRO, {{"d", ".2"}});
    perform_shortest_path_with_graph(cs, gs, ks, sun_pit, SilenceBlock(4));

    cs.stage_macroblock(FileBlock("unless we have exceptional foresight, or we get very lucky,"), 1);
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"d", "1"}});
    cs.stage_macroblock(FileBlock("there's a very high chance that we crash into this dense pit."), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"lines_opacity", ".1"}});
    perform_shortest_path_with_graph(cs, gs, ks, sun, FileBlock("Going back to the start,"));

    auto path = g.shortest_path(ks->copy_board().get_hash(), klotski_solution.get_hash()).second;
    /*for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        EdgeSet& es = p->second.neighbors;
        for(auto& e : es){
            Edge& ed = const_cast<Edge&>(e);
            ed.opacity = .1;
        }
    }*/
    for(Edge* e : path){
        e->opacity = 10;
    }
    cs.stage_macroblock(FileBlock("The only alternative is to walk one of these very fine lines to the other side."), 2);
    cs.render_microblock();
    gs->state_manager.transition(MICRO, {{"d", ".4"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This line is the shortest path to a solution- let's follow it."), 1);
    cs.render_microblock();

    perform_shortest_path_with_graph(cs, gs, ks, klotski_solution, SilenceBlock(15));

    cs.stage_macroblock(SilenceBlock(.5), 2);
    ks->stage_move({'b', 0, 5});
    cs.render_microblock();
    cs.fade_subscene(MICRO, "ks", 0);
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"d", "1"}});
    gs->state_manager.set({{"lines_opacity", "1"}});
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        EdgeSet& es = p->second.neighbors;
        for(auto& e : es){
            Edge& ed = const_cast<Edge&>(e);
            ed.opacity = .1;
        }
    }
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Interestingly, this is not the path chosen by the world record speedsolver..."), 1);
    //TODO
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Was my friend right about the horizontal bar?"), 1);
    auto ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene_fade_in(MACRO, ks_bd_ptr, "ks_bd");
    cs.slide_subscene(MACRO, "ks", -.5, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's go ahead and highlight every node-"), 1);
    gs->next_hash = 0;
    cs.state_manager.transition(MICRO, {{"ks_bd.x",".15"},{"ks_bd.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    ks_bd_ptr->state_manager.transition(MICRO, board_width_height);
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        float b_avg = 0, d_avg = 0;
        for(int x = 0; x < 4; x++) for(int y = 0; y < 5; y++) {
            if(n.data->representation[x+y*4] == 'b') b_avg += y;
            if(n.data->representation[x+y*4] == 'd') d_avg += y;
        }
        b_avg /= 4;
        d_avg /= 2;
        n.color = (b_avg < d_avg + 1 ? 0xff0000 : 0) | (b_avg > d_avg - 1 ? 0x00ff00 : 0);
    }
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0xff0000) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("Red when the bar is under the block,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.6), 3);
    ks_bd_ptr->stage_move({'b', 1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'d', -1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', 0, 1});
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0xffff00) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("yellow when the bar is beside the block,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.6), 2);
    ks_bd_ptr->stage_move({'b', 0, 2});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', -1, 0});
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0x00ff00) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("and green when the block has been moved under the bar."), 1);
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if('b' == n.data->representation[13] && 'b' == n.data->representation[18]) n.color = 0xff00ffff;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("Now once again, take a peek at the solution set..."), 1);
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0x00ff00) n.color |= 0xff000000;
    }
    cs.stage_macroblock(FileBlock("Sure enough, they have an extremely close overlap!"), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"points_opacity", "0"}});
    cs.fade_subscene(MICRO, "ks_bd", 0);
    cs.stage_macroblock(FileBlock("So, my friend's intuition was right."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What else can we learn?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("I'm going to zoom in on a faraway land."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's select a small region of nodes here."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now I'm going to show you the boards for all of those nodes,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but blurred together so that we can only see the shared patterns."), 1);
    cs.render_microblock();
}

void recursive_placer(unordered_set<string>& set, const string& rep, int piece_number){
    if(piece_number == 10) { set.insert(rep); return; }
    int piece_w = 0;
    int piece_h = 0;
    char piece_c = 'a' + piece_number;
    if(piece_number == 0){ // Sun
        piece_w = 2;
        piece_h = 2;
    }
    else if(piece_number == 1){ // Horizontal
        piece_w = 2;
        piece_h = 1;
    }
    else if(piece_number < 6){ // Verticals
        piece_w = 1;
        piece_h = 2;
    }
    else if(piece_number < 10){ // Dots
        piece_w = 1;
        piece_h = 1;
    }
    for(int x = 0; x < 4 - piece_w; x++){
        for(int y = 0; y < 5 - piece_h; y++){
            string child = rep;
            for(int dx = 0; dx < piece_w; dx++) {
                for(int dy = 0; dy < piece_h; dy++) {
                    if(child[x+dx + (y+dy)*4] != '.') {
                        goto next;
                    }
                    child[x+dx + (y+dy)*4] = piece_c;
                }
            }
            recursive_placer(set, child, piece_number+1);
            next: ;
        }
    }
}

void part9(){
    cs.stage_macroblock(FileBlock("Remember how this puzzle has positions that can't be reached?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Klotski is the same way."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("You might notice that although the board is symmetrical in the y axis, "), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("why not the x axis too?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Although you can theoretically rearrange all the pieces upside down,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("there is no valid set of moves that will transition between these two areas."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's add in the second half."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Are there any other disconnected islands?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's add them!"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock(""), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock(""), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock(""), 1);
    cs.render_microblock();

    // Recursively find all of the possible states including those which are unreachable
    Graph omni;
    unordered_set<string> set;
    recursive_placer(set, "....................", 0);
    for(const string& s : set) omni.add_to_stack(new KlotskiBoard(4, 5, s, false));
    omni.sanitize_edges();

    while(set.size() > 0) {
        Graph g;
        auto it = set.begin();

        KlotskiBoard* kb = new KlotskiBoard(4, 5, *it, false); 
        KlotskiBoard* m0 = new KlotskiBoard(4, 5, omni.nodes.find(kb->get_reverse_hash()).second.data.representation, false);
        KlotskiBoard* m1 = new KlotskiBoard(4, 5, omni.nodes.find(kb->get_reverse_hash_2()).second.data.representation, false);
        KlotskiBoard* m2 = new KlotskiBoard(4, 5, omni.nodes.find(m0->get_reverse_hash_2()).second.data.representation, false);
        g.add_to_stack(kb);
        g.add_to_stack(m0);
        g.add_to_stack(m1);
        g.add_to_stack(m2);

        g.expand_completely();

        CompositeScene cs;
        shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
        shared_ptr<LatexScene> ls = make_shared<LatexScene>("Node Count: " + g.size(), 1, .3, .3);
        cs.add_scene(gs, "gs");
        cs.add_scene(ls, "ls", .15, .15);
        cs.stage_macroblock(SilenceSegment(4), 1);
        cs.render_microblock();

        for(Node& n : g.nodes) set.remove(n.data->representation);
    }
}

void part10(){
    // Lists of KlotskiBoards
    auto gpt = {gpt2, gpt3, weird1, weird2, weird3};
    auto other = {apk, mathgames_12_13_04, euler766_easy};
    auto suns = {fatsun, truncatedsun};
    auto geometry = {jam3x3, cube_4d, cube_6d, big_block, diamond, doublering, outer_ring, plus_3_corners, plus_4_corners, ring, ring_big, rows, small_block, t_shapes, triangles, triangles2, manifold_1d, manifold_2d, manifold_3d, manifold_4d, full_15_puzzle};

    showcase_graph(beginner, FileBlock("Just for fun, let's check out some more graphs!"));
    showcase_graph(advanced, FileBlock("Here are some block puzzles I found online."));
    showcase_graph(expert, SilenceBlock(5));
    showcase_graph(reddit, SilenceBlock(5));
    showcase_graph(thinkfun1, SilenceBlock(5));
    showcase_graph(thinkfun2, SilenceBlock(5));
    showcase_graph(thinkfun3, SilenceBlock(5));
    showcase_graph(video, SilenceBlock(5));
    showcase_graph(guh3, SilenceBlock(5));
    showcase_graph(guh4, SilenceBlock(5));

    showcase_graph(sun_no_minis, FileBlock("Here's the Klotski puzzle, but without the small blocks!"));
}

void render_video() {
    //FOR_REAL = false;
    //PRINT_TO_TERMINAL = false;
    bool whole = false;
    if(whole) {
        part1();
        part2();
        part3();
        part4();
        part5();
        part6();
        part7();
    }
    //part8();
    //part9();
    part10();
}
