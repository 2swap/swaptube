#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/PauseScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Media/LoopAnimationScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/Mp4Scene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

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
    {"decay",".92"},
    {"dimensions","2.98"},
    {"surfaces_opacity","0"}, // Whether we want to draw the board at every node
    {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
};
StateSet default_graph_state_chill{
    {"q1", "1"},
    {"qi", "<t> .9 * cos .08 *"},
    {"qj", "<t> .71 * sin .08 *"},
    {"qk", "0"}, // Camera orientation quaternion
    {"decay",".92"},
    {"dimensions","2.98"},
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

void recursive_placer(unordered_set<string>& set, const string& rep, int piece_number, int min_index = -1){
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
    for(int x = 0; x < 4 + 1 - piece_w; x++){
        for(int y = 0; y < 5 + 1 - piece_h; y++){
            int index = x + y * 4;
            if(index <= min_index && (piece_number > 2 && piece_number != 6)) continue; // Dodge transpositions
            string child = rep;
            for(int dx = 0; dx < piece_w; dx++) {
                for(int dy = 0; dy < piece_h; dy++) {
                    if(child[x+dx + (y+dy)*4] != '.') {
                        goto next;
                    }
                    child[x+dx + (y+dy)*4] = piece_c;
                }
            }
            recursive_placer(set, child, piece_number+1, index);
            next: ;
        }
    }
}

void perform_shortest_path_with_graph(CompositeScene& cs, const shared_ptr<GraphScene>& gs_ptr, const shared_ptr<KlotskiScene>& ks_ptr, KlotskiBoard end, const Macroblock &msg) {
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    g.expand();
    auto path = g.shortest_path(ks_ptr->copy_board().get_hash(), end.get_hash()).first;
    cs.stage_macroblock(msg, path.size()-1);
    path.pop_front();
    while(cs.microblocks_remaining()){
        double next = *(path.begin());
        path.pop_front();
        Node node = g.nodes.find(next)->second;
        KlotskiBoard* next_board = dynamic_cast<KlotskiBoard*>(node.data);
        ks_ptr->stage_move(ks_ptr->copy_board().move_required_to_reach(*next_board));
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }
};

void monospeed_path(CompositeScene& cs, const shared_ptr<GraphScene>& gs_ptr, const shared_ptr<KlotskiScene>& ks_ptr, KlotskiBoard end, int col) {
    Graph* g = gs_ptr->expose_graph_ptr();
    auto path = g->shortest_path(ks_ptr->copy_board().get_hash(), end.get_hash()).first;
    int s = path.size()-1;
    path.pop_front();
    for(int i = 0; i < s; i++){
        double next = *(path.begin());
        path.pop_front();
        Node& node = g->nodes.find(next)->second;
        node.color = col;
        KlotskiBoard* next_board = dynamic_cast<KlotskiBoard*>(node.data);
        ks_ptr->stage_move(ks_ptr->copy_board().move_required_to_reach(*next_board));
        cs.render_microblock();
    }
};

void perform_shortest_path(CompositeScene& cs, const shared_ptr<KlotskiScene>& ks_ptr, KlotskiBoard end, const Macroblock &msg) {
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    g.expand();
    auto path = g.shortest_path(ks_ptr->copy_board().get_hash(), end.get_hash()).first;
    ks_ptr->stage_macroblock(msg, path.size()-1);
    path.pop_front();
    while(ks_ptr->microblocks_remaining()){
        double next = *(path.begin());
        path.pop_front();
        Node node = g.nodes.find(next)->second;
        KlotskiBoard* next_board = dynamic_cast<KlotskiBoard*>(node.data);
        ks_ptr->stage_move(ks_ptr->copy_board().move_required_to_reach(*next_board));
        ks_ptr->render_microblock();
    }
};

void showcase_graph(const KlotskiBoard& kb, const Macroblock& mb) {
    CompositeScene cs;
    // Create a KlotskiScene for the given board and set its state
    auto ks_ptr = make_shared<KlotskiScene>(kb);
    ks_ptr->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks_ptr, "ks");
    cs.state_manager.set("ks.opacity", ".5");
    cs.state_manager.set(board_position);

    // Create a graph starting from this board
    Graph g;
    g.add_to_stack(new KlotskiBoard(kb));
    auto gs_ptr = make_shared<GraphScene>(&g, false, 1.2, 1);
    gs_ptr->state_manager.set(default_graph_state);
    gs_ptr->state_manager.set({{"points_opacity", ".3"}, {"lines_opacity", ".2"}, {"decay",".8"}, {"d", "1.3"},  {"physics_multiplier", "100"}});
    cs.add_scene(gs_ptr, "gs", .6, .5, true);

    // Gradually expand the graph to reveal its structure
    int expansion_steps = get_graph_size(kb);
    cs.stage_macroblock(mb, expansion_steps * 2);
    int i = 0;
    while (cs.microblocks_remaining()) {
        i++;
        if(cs.render_microblock() > 0) {
            g.expand(i);
            i = 0;
        }
    }
}

void part0(){
    // Add KlotskiScene and GraphScene for Intermediate puzzle, begin making 50 random moves.
    CompositeScene cs;
    auto ks_ptr = make_shared<KlotskiScene>(sun);
    ks_ptr->state_manager.set(board_width_height);
    Graph g;
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_board()));
    auto gs_ptr = std::make_shared<GraphScene>(&g, true);
    gs_ptr->state_manager.set(default_graph_state);
    cs.add_scene(gs_ptr, "gs");
    cs.add_scene(ks_ptr, "ks");
    cs.state_manager.set(board_position);
    cs.stage_macroblock(FileBlock("This is a random agent exploring the state space of a sliding block puzzle."), 70);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges();
        // Highlight the node of the board on the state-space graph
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("The puzzle is called Klotski."), 2);
    ks_ptr->state_manager.transition(MICRO,{{"w", ".85"},{"h", ".85"}});
    cs.state_manager.transition(MICRO, {{"ks.x",".5"},{"ks.y",".43"}});
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, make_shared<LatexScene>(latex_text("Klotski"), 1, .5, .2), "ls", .5, .9);
    cs.render_microblock();

    showcase_graph(reddit, FileBlock("These block puzzles are quite common,"));
    showcase_graph(expert, FileBlock("but aside from just being puzzles, these sliding blocks define graphs,"));
    showcase_graph(thinkfun2, FileBlock("modeling convoluted topologies with their local substructure,"));
    showcase_graph(thinkfun3, FileBlock("as well as their global superstructure."));
}

void part1(){
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
    cs.stage_macroblock(SilenceBlock(3), 100);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        // Add the new node
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges();
        // Highlight the node of the board on the state-space graph
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    perform_shortest_path_with_graph(cs, gs_ptr, ks_ptr, intermediate, SilenceBlock(.5)); // Go back to start to have a deterministic starting position

    // Set ks.highlight_char to highlight the block which needs to get freed.
    auto ksb_ptr= make_shared<KlotskiScene>(KlotskiBoard(6, 6, "............bb......................", true));
    ksb_ptr->state_manager.set(board_width_height);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.add_scene_fade_in(MICRO, ksb_ptr, "ksb");
    cs.state_manager.set({{"ksb.x",".15"},{"ksb.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MICRO, {{"ks.x",".5"},{"ks.y",".5"}});
    cs.state_manager.transition(MICRO, {{"ksb.x",".5"},{"ksb.y",".5"}});
    cs.fade_subscene(MICRO, "gs", 0);
    ks_ptr->state_manager.transition(MICRO, {{"w","1"},{"h","1"}});
    ksb_ptr->state_manager.transition(MICRO, {{"w","1"},{"h","1"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Often, it's a 6x6 grid, where the solution is to free this block from the hole on the right."), 3);
    cs.fade_subscene(MICRO, "ks", 0);
    cs.render_microblock();
    ksb_ptr->stage_move({'b', 12, 0});
    cs.render_microblock();
    cs.fade_subscene(MICRO, "ks", .5);
    cs.render_microblock();

    cs.fade_subscene(MACRO, "ksb", 0);
    int flashes = 4;
    cs.stage_macroblock(FileBlock("We can't do that yet, since this piece is in the way..."), flashes*2);
    auto ksd= make_shared<KlotskiScene>(KlotskiBoard(6, 6, "..a.....a.....a.....................", true));
    auto ksx= make_shared<KlotskiScene>(KlotskiBoard(6, 6, "....................................", true));
    cs.add_scene(ksd, "ksd");
    cs.add_scene(ksx, "ksx");
    for(int i = 0; i < flashes; i++) {
        cs.state_manager.set("ksd.opacity", "1");
        cs.render_microblock();
        cs.state_manager.set("ksd.opacity", "0");
        cs.render_microblock();
    }
    cs.remove_subscene("ksd");

    cs.stage_macroblock(SilenceBlock(.2), 1);
    cs.fade_subscene(MACRO, "ksx", 0);
    cs.render_microblock();
    cs.remove_subscene("ksx");

    cs.fade_subscene(MACRO, "ks", 1);
    cs.fade_subscene(MICRO, "gs", 1);
    cs.state_manager.transition(MICRO, board_position);
    ks_ptr->state_manager.transition(MICRO, board_width_height);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_subscene("ksx");

    // Delete all nodes of the graph except for the current one. Turn on surface opacity, and turn off edge opacity.
    cs.stage_macroblock(SilenceBlock(3), g.size()-1);
    gs_ptr->state_manager.transition(MACRO, {{"surfaces_opacity",".5"},{"lines_opacity","0"}});
    unordered_map<double,Node> nodes_copy = g.nodes;
    for(auto it = nodes_copy.begin(); it != nodes_copy.end(); ++it){
        double id_here = it->first;
        if(id_here == gs_ptr->curr_hash) continue;
        g.remove_node(id_here);
        cs.render_microblock();
    }
    g.clear_queue();

    // Re-center
    cs.stage_macroblock(FileBlock("Let's represent the current position of the puzzle as a node."), 1);
    cs.render_microblock();

    // Make one move and insert it on the graph.
    ks_ptr->highlight_char = 'h';
    gs_ptr->state_manager.transition(MICRO, {{"physics_multiplier","1"}});
    cs.stage_macroblock(FileBlock("If we make one move on the puzzle,"), 5);
    ks_ptr->stage_move({'h', -1, 0});
    cs.render_microblock();
    ks_ptr->stage_move({'h', 1, 0});
    cs.render_microblock();
    ks_ptr->stage_move({'h', -1, 0});
    cs.render_microblock();
    ks_ptr->stage_move({'h', 1, 0});
    cs.render_microblock();
    ks_ptr->stage_move({'h', -1, 0});
    cs.render_microblock();
    g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
    g.add_missing_edges();
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.stage_macroblock(FileBlock("we arrive at a different node."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Since these positions are just one move apart,"), 1);
    cs.render_microblock();

    // Turn edge opacity on.
    gs_ptr->state_manager.transition(MACRO, {{"lines_opacity","1"}});
    gs_ptr->state_manager.transition(MICRO, {{"physics_multiplier","5"}});
    cs.stage_macroblock(FileBlock("We draw an edge connecting the two nodes."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 2);
    cs.state_manager.transition(MACRO, {{"ks.opacity","0"}});
    ks_ptr->stage_move({'h', 1, 0});
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.render_microblock();
    ks_ptr->stage_move({'h', -1, 0});
    gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
    cs.render_microblock();

    // Make a few random moves.
    gs_ptr->state_manager.transition(MACRO, {{"surfaces_opacity","0"}});
    cs.stage_macroblock(FileBlock("Each node is connected to a few more, forming a labyrinth of paths."), 10);
    while(cs.microblocks_remaining()) {
        ks_ptr->stage_random_move();
        g.add_to_stack(new KlotskiBoard(ks_ptr->copy_staged_board()));
        g.add_missing_edges();
        gs_ptr->next_hash = ks_ptr->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Unhighlight and fade out the KlotskiScene
    gs_ptr->next_hash = 0;
    cs.state_manager.transition(MACRO, {{"gs.opacity","0"}});
    // Expand the graph by one node until it is halfway complete. Fade out everything from the CompositeScene and then delete scenes when faded out.
    int micros = get_graph_size(intermediate) * .7;
    cs.stage_macroblock(FileBlock("You might start to wonder- if we add all the nodes, what would the graph look like?"), micros);
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
        gs2d_ptr->state_manager.set(default_graph_state_chill);
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
        cs.stage_macroblock(FileBlock("such as a grid,"), 1);
        cs.render_microblock();
    }

    {
        // Create new GraphScene for manifold_3d on the right side of the screen and fade it in while expanding the graph completely 
        cs.stage_macroblock(FileBlock("Or maybe it's a total mess!"), 100);
        Graph g3d;
        g3d.add_to_stack(new KlotskiBoard(sun));
        auto gs3d_ptr = make_shared<GraphScene>(&g3d, false, .5, 1);
        gs3d_ptr->state_manager.set(default_graph_state);
        gs3d_ptr->state_manager.set({{"attract","0"}, {"repel","1"}, {"dimensions", "4"}});
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

    if(false){ // Don't do this
        // Fade in and expand a GraphScene for intermediate again, but this time override "physics_multiplier" to be zero so the graph structure is indiscernable.
        cs.stage_macroblock(FileBlock("Maybe it's a dense mesh of interconnected nodes with no grand structure."), 100);
        Graph g_int;
        g_int.add_to_stack(new KlotskiBoard(intermediate));
        auto gs_int_ptr = make_shared<GraphScene>(&g_int, false);
        gs_int_ptr->state_manager.set(default_graph_state);
        gs_int_ptr->state_manager.set({{"attract",".01"}, {"repel","0"}});
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

void part6() {
    // Start over by adding a KlotskiScene.
    shared_ptr<KlotskiScene> ks_ptr = make_shared<KlotskiScene>(sun);

    CompositeScene cs;
    cs.add_scene_fade_in(MICRO, ks_ptr, "ks");
    cs.stage_macroblock(SilenceBlock(.8), 1);
    cs.render_microblock();

    // Make moves according to the shortest path to the position given
    perform_shortest_path(cs, ks_ptr, KlotskiBoard(4, 5, "abbcabbcg.ehj.ehddif", false), FileBlock("Here's Klotski again."));

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

    // Now that dots are back to 0, demonstrate a lateral move. (move piece 'e' left one space)
    ks_ptr->stage_macroblock(FileBlock("so blocks are free to move laterally."), 2);
    ks_ptr->stage_move({'e', -1, 0});
    ks_ptr->render_microblock();
    ks_ptr->stage_move({'e', 1, 0});
    ks_ptr->render_microblock();

    // Show the intermediate puzzle from before.
    cs.remove_all_subscenes();
    cs.add_scene(ks_ptr, "ks");
    cs.state_manager.transition(MACRO, {{"ks.x",".75"}});
    ks_ptr->state_manager.transition(MACRO, {{"w",".5"}});
    cs.stage_macroblock(FileBlock("Compared to the last puzzle, it's _much harder_."), 2);
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, make_shared<KlotskiScene>(intermediate, .5, 1), "ks_intermediate", .25, .5);
    cs.render_microblock();

    // Looping animation scene - me and coworker
    LoopAnimationScene las({"coworker1", "coworker2", "coworker3", 
                            "give1", "give2", "give3",
                            "trying1", "trying2", "trying3",
                            "solved1", "solved2", "solved3",
                            "dizzy1", "dizzy2", "dizzy3"});
    las.state_manager.set({{"loop_length", "3"}});
    las.stage_macroblock(FileBlock("I showed it to a coworker after work,"), 2);
    las.render_microblock();
    las.state_manager.set({{"loop_start", "3"}});
    las.render_microblock();

    // Looping animation scene - coworker struggling to solve
    las.state_manager.set({{"loop_start", "6"}});
    las.stage_macroblock(FileBlock("but he wouldn't leave until he solved it."), 1);
    las.render_microblock();

    // Looping animation scene - coworker dizzy, puzzle solved
    las.stage_macroblock(FileBlock("He finally got it... at 11:00 PM."), 2);
    las.state_manager.set({{"loop_start", "9"}});
    las.render_microblock();
    las.state_manager.set({{"loop_start", "12"}});
    las.render_microblock();
}

void part3(Graph* grt, shared_ptr<KlotskiScene>& tks, shared_ptr<GraphScene>& tgs) {
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
    gs1d->state_manager.set({{"points_opacity", "1"}, {"points_radius_multiplier", "1"}});
    cs.add_scene(gs1d, "gs1d", .6, .5);
    gs1d->next_hash = ks1d->copy_board().get_hash();

    cs.stage_macroblock(FileBlock("To build some intuition, here's a few contrived puzzles first."), get_graph_size(manifold_1d));
    while(cs.microblocks_remaining()) {
        g1d.expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(FileBlock("With just one block, there's only one degree of freedom."), 10);
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

    cs.stage_macroblock(FileBlock("you get the cartesian product of their coordinates, yielding a grid."), get_graph_size(manifold_2d));
    Graph g2d;
    g2d.add_to_stack(new KlotskiBoard(manifold_2d));
    auto gs2d = make_shared<GraphScene>(&g2d, false);
    gs2d->state_manager.set(default_graph_state_chill);
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
    cs.stage_macroblock(FileBlock("Three blocks make a 3d lattice,"), 1);
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
    cs.stage_macroblock(FileBlock("But it gets more fun when the pieces can intersect."), 2);
    cs.state_manager.transition(MICRO, {{"ks4d.x","1"},{"ks4d.y","-1"}});
    cs.state_manager.transition(MICRO, {{"gs4d.x","1"},{"gs4d.y","-1"}});
    cs.state_manager.transition(MACRO, {{"ks2d.x",".15"},{"ks2d.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MACRO, {{"gs2d.x",".6"},{"gs2d.y",".5"}});
    cs.render_microblock();
    cs.render_microblock();

    // Move 2D to left, show ring_big on right
    cs.stage_macroblock(FileBlock("Taking our two-block puzzle,"), 1);
    cs.state_manager.transition(MACRO, {{"ks2d.x",".25"},{"ks2d.y",".25"}});
    cs.state_manager.transition(MACRO, {{"gs2d.x",".25"},{"gs2d.y",".75"}});
    gs2d->state_manager.transition(MACRO, {{"w",".5"},{"h",".5"}});
    ks2d->state_manager.transition(MACRO, {{"w",".5"},{"h",".5"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but with blocks opposing each other,"), 1);
    auto ksr7 = make_shared<KlotskiScene>(ring_7x7);
    ksr7->state_manager.transition(MACRO, {{"w",".5"},{"h",".5"}});
    cs.add_scene_fade_in(MICRO, ksr7, "ksr7");
    cs.state_manager.set({{"ksr7.x",".75"},{"ksr7.y",".25"}});
    Graph gr7;
    gr7.add_to_stack(new KlotskiBoard(ring_7x7));
    gr7.expand();
    auto gsr = make_shared<GraphScene>(&gr7, false, 0.5,0.5);
    gsr->state_manager.set(default_graph_state_chill);
    cs.add_scene_fade_in(MICRO, gsr, "gsr", 0.75, 0.75);
    while(cs.microblocks_remaining()) {
        gr7.expand(1);
        cs.render_microblock();
    }

    // Overlap invalid region
    cs.stage_macroblock(FileBlock("a section of the 2d structure is no longer valid, representing overlapping pieces."), 1);
    cs.state_manager.transition(MACRO, {{"ks2d.x","-.25"}, {"gs2d.x","-.25"}});
    cs.state_manager.transition(MACRO, {{"gsr.x",".6"}, {"gsr.y",".5"}});
    gsr->state_manager.transition(MACRO, {{"w","1"}, {"h","1"}});
    cs.state_manager.transition(MACRO, {{"ksr7.x",".15"}, {"ksr7.y",to_string(yval)}});
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 3);
    ksr7->stage_move({'a', 0, 1});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'b', 1, 0});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'b', 1, 0});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 4);
    ksr7->stage_move({'a', 0, 3});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
    ksr7->stage_move({'a', 0, -3});
    gsr->next_hash = ksr7->copy_staged_board().get_hash();
    cs.render_microblock();
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

    cs.stage_macroblock(FileBlock("With the two blocks on the same lane,"), get_graph_size(triangle));
    tks = make_shared<KlotskiScene>(triangle);
    tks->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, tks, "tks");
    cs.state_manager.set({{"tks.x",".15"},{"tks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    grt->add_to_stack(new KlotskiBoard(triangle));
    tgs = make_shared<GraphScene>(grt, false);
    tgs->state_manager.set(default_graph_state_chill);
    cs.add_scene_fade_in(MICRO, tgs, "tgs", 0.6, 0.5);
    tgs->next_hash = tks->copy_board().get_hash();
    while(cs.microblocks_remaining()) {
        grt->expand(1);
        cs.render_microblock();
    }
    cs.stage_macroblock(FileBlock("we get this triangle shape."), 1);
    cs.render_microblock();

    // Move top then bottom
    tgs->state_manager.transition(MACRO, {{"d","1.5"}});
    cs.stage_macroblock(FileBlock("Wherever the top block is, that serves as a bound for the bottom block."), 35);
    for(int j = 0; j < 5; j++) {
        for(int i=0; i<5-j; ++i){
            tks->stage_move({'c', 0, -1});
            tgs->next_hash = tks->copy_staged_board().get_hash();
            cs.render_microblock();
        }
        for(int i=0; i<5-j; ++i){
            tks->stage_move({'c', 0, 1});
            tgs->next_hash = tks->copy_staged_board().get_hash();
            cs.render_microblock();
        }
        tks->stage_move({'a', 0, 1});
        tgs->next_hash = tks->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    // Mirror-triangle graph
    tgs->state_manager.transition(MACRO, {{"repel",".1"}});
    tgs->state_manager.transition(MACRO, {{"decay",".7"},{"physics_multiplier","4"}});
    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceBlock(2), 21);
    grt->add_to_stack(new KlotskiBoard(triangle_inv));
    tgs->next_hash = tks->copy_staged_board().get_hash();
    while(cs.microblocks_remaining()){
        grt->expand(1);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("With a puzzle like this, we implicitly create an imaginary counterpart structure,"), 6);
    for(int i = 0; i < 6; i++) {
        const std::string s = std::string(9*i, '.') + "....cx.......c........a........a...." + std::string((5 - i)*9, '.');
        string s_dot = s;
        replace(s_dot.begin(), s_dot.end(), 'x', '.');
        string s_c = s;
        replace(s_c.begin(), s_c.end(), 'x', 'c');
        KlotskiBoard a(9, 9, s_dot, true);
        KlotskiBoard b(9, 9, s_c, true);
        double ah = a.get_hash();
        double bh = b.get_hash();
        grt->add_directed_edge(ah, bh, 0);
        grt->add_directed_edge(bh, ah, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("corresponding to the valid states which are unreachable without one block passing through the other."), 3);
    cs.render_microblock();
    cs.render_microblock();

    tks->stage_move({'c', 0, -4});
    KlotskiBoard lie(9, 9, "...............................cc.......c........a........a......................", true );
    tgs->next_hash = lie.get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    lie = lie.move_piece({'c', 0, 4});
    tks->stage_move({'c', 0, 4});
    tgs->next_hash = tks->copy_staged_board().get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    // 3-intersecting blocks example
    auto ks3rb = make_shared<KlotskiScene>(iblock);
    ks3rb->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks3rb, "ks3rb");
    cs.state_manager.set({{"ks3rb.x",".15"},{"ks3rb.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("3 intersecting pieces still form a cube, there's just some excavated areas."), get_graph_size(iblock));
    Graph g3rb;
    g3rb.add_to_stack(new KlotskiBoard(iblock));
    auto gs3rb = make_shared<GraphScene>(&g3rb, false);
    gs3rb->state_manager.set(default_graph_state);
    gs3rb->state_manager.transition(MACRO, {{"d", ".65"}});
    cs.add_scene(gs3rb, "gs3rb", 0.6, 0.5);
    while(cs.microblocks_remaining()){
        g3rb.expand(1);
        cs.render_microblock();
    }
    gs3rb->state_manager.transition(MACRO, {{"d", "1"}});
    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    // Fade out all
    cs.stage_macroblock(FileBlock("But as the piece number gets higher, the graph's dimensionality has less to do with the number of pieces, and more to do with the number of empty spaces."), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
}

void part4() {
    CompositeScene cs;

    // apk puzzle and full expansion
    auto ks_apk = make_shared<KlotskiScene>(apk);
    ks_apk->state_manager.set(board_width_height);
    Graph g_apk;
    g_apk.add_to_stack(new KlotskiBoard(apk));
    auto gs_apk = make_shared<GraphScene>(&g_apk, false);
    gs_apk->state_manager.set(default_graph_state);
    gs_apk->state_manager.set({{"dimensions", "3"}});
    cs.add_scene(gs_apk, "gs_apk", 0.6, 0.5);
    cs.add_scene(ks_apk, "ks_apk");
    cs.state_manager.set({{"ks_apk.x","-.15"},{"ks_apk.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MACRO, {{"ks_apk.x",".15"},{"ks_apk.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As an example, this puzzle has some cool behavior."), get_graph_size(apk));
    while(cs.microblocks_remaining()){
        g_apk.expand(1);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("If I expand it out entirely,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Notice that it has some overall superstructure,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.5), 1);
    gs_apk->next_hash = ks_apk->copy_board().get_hash();
    cs.render_microblock();

    gs_apk->state_manager.transition(MACRO, {{"d", "1"}});
    unordered_map<double,Node> nodes_copy = g_apk.nodes;
    list<double> to_remove;
    for(auto it = nodes_copy.begin(); it != nodes_copy.end(); ++it){
        double id_here = it->first;
        if(g_apk.dist(id_here, ks_apk->copy_board().get_hash()) < 20) continue;
        to_remove.push_back(id_here);
    }
    g_apk.clear_queue();

    cs.stage_macroblock(FileBlock("but also, in a small portion of the graph,"), to_remove.size());
    for(double d : to_remove){
        g_apk.remove_node(d);
        cs.render_microblock();
    }

    gs_apk->state_manager.transition(MICRO, {{"dimensions", "2.99"}, {"d", "1"}});
    cs.stage_macroblock(FileBlock("the local behavior is quite nicely patterned as well."), 1);
    cs.render_microblock();

    gs_apk->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("It's like a cute little local euclidean manifold with two dimensions- two degrees of freedom."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's because on the puzzle, there are two holes allowing movement."), 4);
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("1", 1, .23, .23);
    //ls->state_manager.transition(MICRO, {{"w",".25"},{"h",".25"}});
    cs.add_scene(ls, "ls", .18, .105);
    cs.fade_subscene(MICRO, "ls", 0);
    cs.render_microblock();
    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("2", 1, .23, .23);
    //ls2->state_manager.transition(MICRO, {{"w",".25"},{"h",".25"}});
    cs.add_scene(ls2, "ls2", .238, .42);
    cs.fade_subscene(MICRO, "ls2", 0);
    cs.render_microblock();
    cs.remove_subscene("ls");
    cs.remove_subscene("ls2");

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
    gs_15->state_manager.set(default_graph_state_chill);
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
    gs2d->state_manager.set(default_graph_state_chill);
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

void part5() {
    CompositeScene cs;

    // intermediate graph overlay
    cs.stage_macroblock(FileBlock("Here's one of the puzzles we started with."), 1/*get_graph_size(intermediate)*/);
    auto ks_int = make_shared<KlotskiScene>(intermediate);
    //ks_int->state_manager.set(board_width_height);
    cs.add_scene_fade_in(MICRO, ks_int, "ks_int");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Its superstructure is quite simple."), 1);
    cs.render_microblock();

    // pause scene
    cs.stage_macroblock(FileBlock("Pause for a moment to think through what it might look like. You might be able to guess its form from the arrangement of the pieces!"), 1);
    vector<const KlotskiBoard*> boards = {&weird1, &euler766_easy, &beginner, &diamond};
    vector<string> names = {"w1","eul","beg","dia"};
    vector<Graph*> graphs;
    for(int i=0;i<boards.size();++i){
        Graph* g = new Graph;
        graphs.push_back(g);
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
        cs.add_scene_fade_in(MICRO, gs, names[i], top?.25:.75, left?.25:.75, .5, true);
    }
    shared_ptr<PauseScene> ps = make_shared<PauseScene>();
    cs.add_scene(ps, "ps");
    cs.render_microblock();
    cs.remove_subscene("ps");

    cs.fade_all_subscenes_except(MICRO, "ks_int", 0);
    cs.stage_macroblock(FileBlock("Ready?"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("ks_int");
    for(Graph* g : graphs) delete g;

    perform_shortest_path(cs, ks_int, KlotskiBoard(6, 6, "..fffc..a..cbba...dda..e.....e..hhhe", true), SilenceBlock(1));

    cs.stage_macroblock(SilenceBlock(.7), 1);
    cs.render_microblock();

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
    tds.stage_macroblock(FileBlock("From this perspective, the puzzle is almost symmetrical."), 5);
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
    cs.add_scene(bds, "bds");
    cs.stage_macroblock(FileBlock("The key is recognizing that these two pieces stay latched in one of two spots."), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ks_int", 1);
    cs.stage_macroblock(FileBlock("They can either be to the left of the vertical red bar,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.3), 1);
    cs.fade_subscene(MICRO, "bds", 1);
    cs.fade_subscene(MICRO, "ks_int", 0);
    cs.render_microblock();
    cs.remove_subscene("ks_int");

    cs.stage_macroblock(SilenceBlock(.5), 2);
    bds->stage_move({'b', 3, 0});
    cs.render_microblock();
    bds->stage_move({'d', 3, 0});
    cs.render_microblock();

    ks_int = make_shared<KlotskiScene>(KlotskiBoard(6,6,"..fffc..a..c..abb...adde.....e..hhhe",true));
    cs.fade_subscene(MICRO, "bds", 0);
    cs.add_scene_fade_in(MICRO, ks_int, "ks_int");
    cs.stage_macroblock(FileBlock("or they can be to the right of it."), 1);
    cs.render_microblock();
    cs.remove_subscene("bds");

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

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.state_manager.transition(MACRO, {{"ks_int.x",".15"},{"ks_int.y",to_string(yval)}});
    ks_int->state_manager.transition(MACRO, board_width_height);
    cs.stage_macroblock(FileBlock("This is the defining characteristic of the graph's superstructure."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), get_graph_size(intermediate));
    Graph g_int;
    g_int.add_to_stack(new KlotskiBoard(intermediate));
    auto gs_int = make_shared<GraphScene>(&g_int, false);
    gs_int->state_manager.set(default_graph_state_chill);
    cs.add_scene(gs_int, "gs_int", .6, .5);
    while(cs.microblocks_remaining()) {
        g_int.expand(1);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("It's a square connecting 4 corners."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can color the nodes on the graph in correspondence with the position of the puzzle."), 1);
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
    gs_int->state_manager.transition(MACRO, {{"d", "1.15"}});
    gs_int->state_manager.transition(MACRO, {{"lines_opacity",".1"}, {"points_radius_multiplier","1.5"}});
    cs.state_manager.transition(MACRO, {{"gs_int.x", ".7"}, {"ks_int.x",".25"}, {"ks_int.y",".5"}});
    ks_int->state_manager.transition(MACRO, {{"w",".4"}, {"h","1"}});
    gs_int->next_hash = ks_int->copy_staged_board().get_hash();
    cs.render_microblock();

    perform_shortest_path_with_graph(cs, gs_int, ks_int, intermediate, SilenceBlock(5));
    KlotskiBoard swaparound(6, 6, "..afff..a..c..abbc.dd..e.....e..hhhe", true );
    perform_shortest_path_with_graph(cs, gs_int, ks_int, swaparound, SilenceBlock(5));
    KlotskiBoard swaparound2(6, 6, "..afff..a..cbba..c...dde.....e..hhhe", true );
    perform_shortest_path_with_graph(cs, gs_int, ks_int, swaparound2, SilenceBlock(5));

    gs_int->state_manager.transition(MACRO, {{"lines_opacity","0"}});
    cs.stage_macroblock(FileBlock("Now, from here, we can highlight all the solutions, where the green block is by the hole-"), 1);
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

    gs_int->state_manager.transition(MACRO, {{"lines_opacity",".02"}});
    perform_shortest_path_with_graph(cs, gs_int, ks_int, intermediate_solution, SilenceBlock(5));

    int flashes = 8;
    cs.stage_macroblock(FileBlock("Unsurprisingly, they all live in the corners with the green block right of the bar."), flashes*2);
    for(int i = 0; i < flashes; i++) {
        for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
            Node& n = p->second;
            n.color = 0x22888888;
            if('b' == n.data->representation[16]) n.color |= 0xffff0000;
            if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
            if('d' == n.data->representation[22]) n.color |= 0x0000ff00;
            if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
        }
        cs.render_microblock();
        for(auto p = g_int.nodes.begin(); p != g_int.nodes.end(); p++){
            Node& n = p->second;
            n.color = 0x22888888;
            if('b' == n.data->representation[17]) n.color |= 0xffff0000;
            if(n.data->representation[12] == 'b') n.color &= 0xff00ffff;
            if('d' == n.data->representation[22]) n.color |= 0x0000ff00;
            if(n.data->representation[18] == 'd') n.color &= 0xffff00ff;
        }
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("Can you figure out why the graph has a higher dimensionality near the corners?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.8), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
}

void part7(shared_ptr<GraphScene>& tgs, shared_ptr<KlotskiScene>& tks) {
    CompositeScene cs;
    // Transition to subpuzzle containing only blocks b and d
    shared_ptr<KlotskiScene> ks = make_shared<KlotskiScene>(sun);
    ks = make_shared<KlotskiScene>(sun);
    cs.add_scene(ks, "ks", .5, -.5);
    cs.fade_all_subscenes(MICRO, 0);
    auto ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene(ks_bd_ptr, "ks_bd");
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_subscene("ks");

    // Animate big piece going under small piece.
    cs.stage_macroblock(FileBlock("He thought the hardest part was getting the box under the horizontal bar."), 6);
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
    cs.add_scene_fade_in(MICRO, ks, "ks", 0.5, 0.5);
    cs.stage_macroblock(FileBlock("Instead of solving it myself,"), 1);
    cs.render_microblock();

    // Start to grow a graph (a hundred nodes or so) in the background
    cs.stage_macroblock(FileBlock("I was more interested to see how it works under the hood."), 100);
    Graph g;
    g.add_to_stack(new KlotskiBoard(sun));
    auto gs = make_shared<GraphScene>(&g, false);
    gs->state_manager.set(default_graph_state);
    cs.add_scene(gs, "gs", .5, .5, true);
    while(cs.microblocks_remaining()){
        g.expand(1);
        cs.render_microblock();
    }

    ks->state_manager.transition(MACRO, board_width_height);
    cs .state_manager.transition(MACRO, {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.stage_macroblock(FileBlock("Is getting the box under the bar actually the hardest part?"), 100);
    while(cs.microblocks_remaining()){
        g.expand(1);
        cs.render_microblock();
    }

    gs->state_manager.transition(MACRO, {{"mirror_force", ".01"}, {"points_radius_multiplier", ".3"}, {"physics_multiplier", "80"}, {"dimensions", "4"}, {"d", ".8"}});
    cs.stage_macroblock(FileBlock("The structure defined by this puzzle- what is its _form_?"), 100);
    while(cs.microblocks_remaining()){
        g.expand(1);
        cs.render_microblock();
    }

    int x = get_graph_size(sun);
    float ii = 3;
    while (g.size() < x) {
        int num_nodes_to_add = ii*ii / 7;
        g.expand(num_nodes_to_add);
        cs.stage_macroblock(SilenceBlock(0.033333), 1);
        cs.render_microblock();
        ii+=.2;
        cout << to_string(g.size()) << endl;
    }

    cs.stage_macroblock(SilenceBlock(15), 2);
    gs->state_manager.transition(MICRO, {{"physics_multiplier", "40"}, {"d", ".5"}, {"dimensions", "3.99"}});
    cs.render_microblock();
    gs->state_manager.transition(MICRO, {{"physics_multiplier", "0"}, {"d", ".9"}, {"dimensions", "3"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's 25,955 nodes."), 3);
    gs->state_manager.transition(MICRO, {{"lines_opacity", "0"}, {"points_opacity", "1"}});
    cs.render_microblock();
    cs.render_microblock();
    gs->state_manager.transition(MICRO, {{"lines_opacity", "1"}, {"points_opacity", "0"}});
    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}});
    cs.render_microblock();
    gs->state_manager.set({{"points_opacity", "1"}});
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x00000000;
    }

    gs->state_manager.transition(MACRO, {{"d", ".5"}, {"lines_opacity", ".3"}, {"points_radius_multiplier","2"}});
    gs->next_hash = sun.get_hash();
    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("The puzzle is symmetrical, so the graph is too.")), 1);
    cs.render_microblock();

    KlotskiBoard side(4, 5, "bbacbbacehddeh..fgij", false);
    KlotskiBoard side_flip(4, 5, "cabbcabbddhe..hejigf", false);
    cs.state_manager.transition(MACRO, {{"ks.x",".5"}});
    perform_shortest_path_with_graph(cs, gs, ks, side, FileBlock("For example, if we take this position,"));

    shared_ptr<KlotskiScene> ks_clone = make_shared<KlotskiScene>(side);
    g.nodes.find(side.get_hash())->second.color = 0xffff0000;
    ks_clone->state_manager.set(board_width_height);
    cs.add_scene(ks_clone, "ks_clone", 0, 0, true);
    cs.state_manager.set({{"ks_clone.x",".5"},{"ks_clone.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.state_manager.transition(MACRO, {{"ks_clone.x",".85"},{"ks_clone.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    cs.state_manager.transition(MACRO, {{"ks.x",".15"},{"ks.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    perform_shortest_path_with_graph(cs, gs, ks_clone, side_flip, FileBlock("the board is a mirror reflection of this node on the opposite side."));

    gs->state_manager.transition(MACRO, {{"points_opacity", "0"}});
    cs.stage_macroblock(SilenceBlock(.8), 1);
    cs.render_microblock();

    g.nodes.find(side.get_hash())->second.color = 0;
    gs->state_manager.transition(MICRO, {{"d", "1"}});
    cs.fade_all_subscenes_except(MICRO, "gs", 0);
    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", ".5"}, {"qk", "0"}});
    gs->next_hash = 0;
    gs->state_manager.set({{"points_radius_multiplier", "0"}, {"points_opacity","1"}});

    gs->state_manager.transition(MICRO, {{"points_radius_multiplier", "1"}});
    cs.remove_all_subscenes_except("gs");
    cs.render_microblock();
    double leftboard = KlotskiBoard(4, 5, "f.giaddcabbcebbhe.jh", false).get_hash();
    double leftboard2 = KlotskiBoard(4, 5, "a.ehafehcddgcbbijbb.", false).get_hash();
    double leftboard3 = KlotskiBoard(4, 5, ".a.cfagcibbehbbehddj", false).get_hash();
    double leftboard4 = KlotskiBoard(4, 5, "a.c.agcfhbbihbbejdde", false).get_hash();
    double leftboard6 = KlotskiBoard(4, 5, "fgbbacbbacijehddeh..", false).get_hash();
    double leftboard7 = KlotskiBoard(4, 5, "bbgfbbcaijcaddhe..he", false).get_hash();
    int col_left = 0xff0088ff;
    int col_right = 0xffff0000;
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){ p->second.color = col_right; }
    unordered_set<double> hashes_l = g.get_neighborhood(leftboard, 85);
    unordered_set<double> hashes_l2 = g.get_neighborhood(leftboard2, 60);
    unordered_set<double> hashes_l3 = g.get_neighborhood(leftboard3, 62);
    unordered_set<double> hashes_l4 = g.get_neighborhood(leftboard4, 62);
    unordered_set<double> hashes_l5 = g.get_neighborhood(klotski_necklace.get_hash(), 52);
    unordered_set<double> hashes_l6 = g.get_neighborhood(leftboard6, 62);
    unordered_set<double> hashes_l7 = g.get_neighborhood(leftboard7, 62);
    for(double d : hashes_l ){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l2){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l3){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l4){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l5){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l6){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l7){ g.nodes.find(d)->second.color = col_left; }
    cs.stage_macroblock(FileBlock("Turning 90 degrees, we see a rough division into two halves."), 2);
    cs.render_microblock();
    gs->next_hash = sun.get_hash();
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"d", ".01"}, {"points_radius_multiplier","1.5"}});
    gs->state_manager.transition(MACRO, less_spinny);
    ks = make_shared<KlotskiScene>(sun);
    cs.add_scene_fade_in(MICRO, ks, "ks");
    ks->state_manager.set(board_width_height);
    cs.state_manager.set(board_position);
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x00000000;
    }
    cs.stage_macroblock(FileBlock("This red node right here is the starting position."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"lines_opacity", ".2"}});
    gs->state_manager.transition(MACRO, {{"d", "1.5"}});
    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", ".5"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("And these are all the solutions- where the square is at the bottom."), 1);
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x00000000;
        if('b' == n.data->representation[13] && 'b' == n.data->representation[18]) n.color |= 0xff00ffff;
    }
    cs.render_microblock();
    gs->state_manager.set({{"points_opacity", "1"}});

    gs->state_manager.transition(MACRO, less_spinny);
    cs.stage_macroblock(FileBlock("All the solution nodes are on the opposite half of the graph as the starting position."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    gs->state_manager.transition(MACRO, {{"d", ".05"}});
    cs.render_microblock();

    const int num_random_moves = 40;
    gs->state_manager.transition(MACRO, {{"d", ".2"}});
    cs.stage_macroblock(FileBlock("By making random moves,"), num_random_moves);
    for (int i = 0; i < num_random_moves; i++) {
        ks->stage_random_move();
        gs->next_hash = ks->copy_staged_board().get_hash();
        cs.render_microblock();
    }

    gs->state_manager.transition(MACRO, {{"d", ".4"}});
    perform_shortest_path_with_graph(cs, gs, ks, sun_pit, SilenceBlock(4));

    gs->state_manager.transition(MICRO, {{"d", "1"}});
    cs.stage_macroblock(FileBlock("unless we get lucky, we're probably gonna crash into this dense pit."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"lines_opacity", ".05"}});
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
        e->opacity = 20;
    }
    cs.stage_macroblock(FileBlock("The only alternative is to walk one of these very fine lines to the other side."), 2);
    cs.render_microblock();
    gs->state_manager.transition(MICRO, {{"d", ".4"}});
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", ".5"}, {"qj", ".1 <t> .2 * cos *"}, {"qk", ".1"}, });
    cs.stage_macroblock(FileBlock("This line is the shortest path to a solution- let's follow it."), 1);
    cs.render_microblock();

    perform_shortest_path_with_graph(cs, gs, ks, klotski_solution, SilenceBlock(15));

    cs.stage_macroblock(SilenceBlock(.5), 2);
    //ks->stage_move({'b', 0, 5});
    cs.render_microblock();
    cs.render_microblock();

    gs->state_manager.transition(MICRO, {{"d", "1"}});
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_subscene("ks");

    CompositeScene wr_with_creds;
    shared_ptr<Mp4Scene> ms = make_shared<Mp4Scene>("WR.mp4");
    shared_ptr<LatexScene> ls = make_shared<LatexScene>(latex_color(OPAQUE_BLACK, latex_text("Lim Kai Yi")), 1, .3, .2);
    shared_ptr<LatexScene> ls2= make_shared<LatexScene>(latex_color(OPAQUE_BLACK, latex_text("3.993 Guinness WR solve")), 1, .3, .2);
    shared_ptr<LatexScene> ls3= make_shared<LatexScene>(latex_color(OPAQUE_BLACK, latex_text("YT: \\@limkaiyiworldrecord")), 1, .3, .1);
    wr_with_creds.add_scene(ms, "ms");
    wr_with_creds.add_scene(ls, "ls", .25, .06);
    wr_with_creds.add_scene(ls2, "ls2", .25, .125);
    wr_with_creds.add_scene(ls3, "ls3", .25, .18);
    gs->next_hash = 0;
    wr_with_creds.stage_macroblock(CompositeBlock(FileBlock("Interestingly, this is not the path used by the guinness world record speedsolver,"), SilenceBlock(.9)), 1);
    wr_with_creds.render_microblock();

    gs->state_manager.transition(MACRO, less_spinny);
    cs.remove_subscene("ks");
    ks = make_shared<KlotskiScene>(sun);
    cs.add_scene(ks, "ks");
    ks->state_manager.set(board_width_height);
    cs.state_manager.set(board_position);
    cs.stage_macroblock(CompositeBlock(FileBlock("who instead takes this line..."), SilenceBlock(4)), 120);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"abbcabbceddhe.fhgij.",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"abbcabbcdd..feghiejh",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"abbcabbcf.eh.gehddij",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fbbagbbaceihce.hdd.j",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fbbagbbachiechje..dd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fbbagbba..iechjechdd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,".f.agbbaibbcehjcehdd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fgiacbbacbbeh..ehjdd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,".facegaceibbh.bbhjdd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"a.fcag.cebbhebbhijdd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"acf.ac.gbbehbbehijdd",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"..fgacehacehbbddbbij",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fg..acehacehbbddbbij",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fgacehacehddbbi.bb.j",false), 0xffffff00);
    monospeed_path(cs, gs, ks, KlotskiBoard(4,5,"fgacehacehdd.bbi.bbj",false), 0xffffff00);

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cs.render_microblock();

    for(Edge* e : path){
        e->opacity = 1;
    }
    cs.fade_subscene(MICRO, "ks", 0);
    gs->state_manager.transition(MICRO, {{"points_opacity", "0"}});
    cs.stage_macroblock(FileBlock("So, was my friend right about the horizontal bar?"), 1);
    ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene_fade_in(MACRO, ks_bd_ptr, "ks_bd");
    cs.state_manager.set({{"ks_bd.x",".15"},{"ks_bd.y",to_string(.15*VIDEO_WIDTH/VIDEO_HEIGHT)}});
    ks_bd_ptr->state_manager.transition(MICRO, board_width_height);
    cs.render_microblock();
    cs.remove_subscene("ks");

    gs->state_manager.transition(MICRO, less_spinny);
    cs.stage_macroblock(FileBlock("Let's highlight every node-"), 1);
    gs->next_hash = 0;
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
             if(b_avg < d_avg - 1) n.color = 0xff0000;
        else if(b_avg < d_avg    ) n.color = 0xff8800;
        else if(b_avg < d_avg + 1) n.color = 0xffff00;
        else if(b_avg < d_avg + 9) n.color = 0x00ff00;
    }
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0xff0000) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    gs->state_manager.set({{"points_opacity", "1"}});
    cs.stage_macroblock(FileBlock("Red when the block is above the bar,"), 1);
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
        if(n.color == 0xff8800) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("orange when they're side by side,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.6), 1);
    ks_bd_ptr->stage_move({'b', 0, 1});
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0xffff00) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("yellow when the block is almost under,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.6), 2);
    ks_bd_ptr->stage_move({'b', 0, 1});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', -1, 0});
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if(n.color == 0x00ff00) n.color |= 0xff000000;
        else n.color &= 0x00ffffff;
    }
    cs.stage_macroblock(FileBlock("and green when the block is finally below the bar."), 1);
    cs.render_microblock();

    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        if('b' == n.data->representation[13] && 'b' == n.data->representation[18]) n.color = 0xff00ffff;
        else n.color &= 0x00ffffff;
    }
    cs.fade_subscene(MICRO, "ks_bd", 0);
    cs.stage_macroblock(FileBlock("Now once again, take a peek at the solution set..."), 1);
    cs.render_microblock();

    gs->state_manager.set({{"highlight_point_opacity", "0"}});
    gs->next_hash = klotski_solution.get_hash();
    gs->state_manager.transition(MICRO, {{"d", ".6"}});
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    int flashes = 5;
    cs.stage_macroblock(FileBlock("There's an extremely close overlap!"), flashes*2);
    for(int i = 0; i < flashes; i++) {
        for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
            Node& n = p->second;
            if(n.color == 0xff00ff00) n.color &= 0x00ffffff;
        }
        cs.render_microblock();
        for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
            Node& n = p->second;
            if(n.color == 0x0000ff00) n.color |= 0xff000000;
        }
        cs.render_microblock();
    }

    gs->state_manager.transition(MICRO, {{"d", "1"}});
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"points_opacity", "0"}});
    cs.fade_subscene(MICRO, "ks_bd", 0);
    cs.stage_macroblock(FileBlock("So, my friend's intuition was right."), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"q1", "1"}, {"qi", "0"}, {"qj", ".5"}, {"qk", "0"}, });
    cs.slide_subscene(MACRO, "gs", -.25, 0);
    gs->next_hash = 0;
    cs.stage_macroblock(CompositeBlock(FileBlock("What else can we learn?"), SilenceBlock(1)), 2);
    cs.render_microblock();
    tgs->next_hash = tgs->curr_hash = 0;
    cs.add_scene(tks, "tks", 1.85, yval);
    cs.add_scene(tgs, "tgs", 1.85, .75);
    cs.state_manager.transition(MACRO, {{"tks.x",".85"},{"tks.y",to_string(yval)}});
    tks->state_manager.set(board_width_height);
    tgs->state_manager.set({{"w",".85"},{"h",".75"}});
    cs.slide_subscene(MACRO, "tgs", -1, 0);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, {{"lines_opacity", ".1"}});
    cs.stage_macroblock(FileBlock("Remember how this puzzle has unreachable positions?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("So does Klotski!"), 1);
    cs.slide_subscene(MACRO, "gs", .25, 0);
    cs.slide_subscene(MACRO, "tks", 1, 0);
    cs.slide_subscene(MACRO, "tgs", 1, 0);
    cs.state_manager.transition(MACRO, board_position);
    cs.render_microblock();
    cs.remove_subscene("tks");
    cs.remove_subscene("tgs");

    gs->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}, });
    cs.stage_macroblock(FileBlock("You might notice that although the graph is horizontally symmetrical, "), 1);
    cs.render_microblock();

    shared_ptr<KlotskiScene> ks_start = make_shared<KlotskiScene>(sun);
    shared_ptr<KlotskiScene> ks_no_b = make_shared<KlotskiScene>(sun_no_b);
    cs.stage_macroblock(FileBlock("why not vertically too?"), 2);
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, ks_start, "ks_start");
    cs.render_microblock();

    shared_ptr<ThreeDimensionScene> tds = make_shared<ThreeDimensionScene>();
    cs.remove_subscene("ks_start");
    tds->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, .5, 0), "ks_start"), ks_start);
    cs.add_scene(tds, "tds");
    cs.stage_macroblock(FileBlock("Although _in theory_ you could rearrange all the pieces upside down,"), 5);
    cs.render_microblock();
    cs.render_microblock();
    tds->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "1"}});
    cs.render_microblock();
    cs.render_microblock();
    tds->state_manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}});
    cs.render_microblock();
    cs.remove_subscene("tds");
    tds->remove_surface("ks_start");
    cs.add_scene(ks_start, "ks_start");

    cs.add_scene(ks_no_b, "ks_no_b");
    cs.stage_macroblock(FileBlock("it's not actually possible,"), 1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ks_start", 0);
    cs.stage_macroblock(FileBlock("unless you take a piece out,"), 1);
    cs.render_microblock();
    cs.remove_subscene("ks_start");

    cs.stage_macroblock(SilenceBlock(1.5), 15);
    ks_no_b->stage_move({'j', -1, 0});
    cs.render_microblock();
    ks_no_b->stage_move({'i', 1, 0});
    cs.render_microblock();
    ks_no_b->stage_move({'e', 0, 1});
    cs.render_microblock();
    ks_no_b->stage_move({'h', 0, 1});
    cs.render_microblock();
    ks_no_b->stage_move({'d', 1, 0});
    cs.render_microblock();
    ks_no_b->stage_move({'f', 0, -3});
    cs.render_microblock();
    ks_no_b->stage_move({'i', 0, -3});
    cs.render_microblock();
    ks_no_b->stage_move({'d', -2, 0});
    cs.render_microblock();
    ks_no_b->stage_move({'g', 0, -3});
    cs.render_microblock();
    ks_no_b->stage_move({'j', 0, -3});
    cs.render_microblock();
    ks_no_b->stage_move({'d', 1, 0});
    cs.render_microblock();
    ks_no_b->stage_move({'a', 0, 1});
    cs.render_microblock();
    ks_no_b->stage_move({'c', 0, 1});
    cs.render_microblock();
    ks_no_b->stage_move({'g', 1, 0});
    cs.render_microblock();
    ks_no_b->stage_move({'f', -1, 0});
    cs.render_microblock();

    shared_ptr<KlotskiScene> flip = make_shared<KlotskiScene>(klotski_flip);
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0xffffffff;
    }
    cs.add_scene_fade_in(MICRO, flip, "flip", 0.5, 0.5, 1);
    cs.stage_macroblock(FileBlock("and put it back in."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    //gs->state_manager.transition(MACRO, {{"points_radius_multiplier",".5"}, {"points_opacity", "1"}});
    cs.fade_all_subscenes_except(MICRO, "gs", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("gs");

    gs->state_manager.set({{"physics_multiplier", "500"}});
    gs->state_manager.transition(MACRO, {{"mirror_force", ".01"}, {"physics_multiplier", "100"}, {"decay",".9"}, {"dimensions", "3.999"}});
    cs.stage_macroblock(FileBlock("Adding the flipped board and all of the other positions reachable, we get our doubly-symmetric graph."), 260);
    g.add_to_stack(new KlotskiBoard(klotski_flip));
    for (int i = 0; i < 260; i++) {
        g.expand(100);
        cs.render_microblock();
    }

    gs->state_manager.transition(MACRO, {{"points_radius_multiplier","0"}});
    cs.stage_macroblock(SilenceBlock(4), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Are there any other disconnected islands?"), 1);
    cs.render_microblock();

    unordered_set<string> set;
    recursive_placer(set, "....................", 0);
    cout << "Set size: " + to_string(set.size()) << endl;
    for(const string& s : set) {
        g.add_to_stack(new KlotskiBoard(4, 5, s, false));
    }
    g.expand(-1);

    cs.stage_macroblock(FileBlock("It turns out, quite a lot!"), 1);
    gs->next_hash = klotski_bonus.get_hash();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    gs->state_manager.transition(MACRO, {{"points_radius_multiplier","1"}});
    gs->state_manager.transition(MACRO, {{"d", ".08"}});
    cs.render_microblock();

    ks = make_shared<KlotskiScene>(klotski_bonus);
    cs.add_scene(ks, "ks");
    cs.state_manager.set(board_position);
    ks->state_manager.set(board_width_height);
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){
        Node& n = p->second;
        n.color = 0x00000000;
    }
    gs->state_manager.set({{"highlight_point_opacity", "1"},{"points_opacity", "1"}});
    gs->state_manager.transition(MACRO, {{"d", ".03"}, {"physics_multiplier", "0"}});
    cs.stage_macroblock(FileBlock("Here's the biggest one, with only 248 nodes."), 1);
    cs.render_microblock();

    gs->state_manager.transition(MACRO, less_spinny);
    perform_shortest_path_with_graph(cs, gs, ks, klotski_bonus2, FileBlock("It's primarily oriented as a long path, with occasional branching."));

    gs->state_manager.transition(MACRO, {{"d", "1"}, {"highlight_point_opacity", "0"}});
    cs.stage_macroblock(SilenceBlock(3), 2);
    cs.render_microblock();
    gs->next_hash = 0;
    cs.render_microblock();

    gs->next_hash = sun.get_hash();
    cs.stage_macroblock(FileBlock("So, what makes this puzzle hard?"), 1);
    cs.fade_subscene(MICRO, "ks", 0);
    cs.render_microblock();
    cs.remove_subscene("ks");

    ks_bd_ptr = make_shared<KlotskiScene>(KlotskiBoard(4, 5, ".bb..bb..dd.........", false));
    cs.add_scene_fade_in(MACRO, ks_bd_ptr, "ks_bd");
    cs.stage_macroblock(FileBlock("We could say, for example,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("that it's hard because getting the block under the bar is hard,"), 4);
    ks_bd_ptr->stage_move({'b', 1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'d', -1, 0});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', 0, 3});
    cs.render_microblock();
    ks_bd_ptr->stage_move({'b', -1, 0});
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ks_bd", 0);
    unordered_set<double> hashes2 = g.get_neighborhood(leftboard, 250);
    for(double d : hashes2){ g.nodes.find(d)->second.color = col_right; }
    for(double d : hashes_l ){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l2){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l3){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l4){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l5){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l6){ g.nodes.find(d)->second.color = col_left; }
    for(double d : hashes_l7){ g.nodes.find(d)->second.color = col_left; }
    cs.stage_macroblock(FileBlock("or that it's hard because crossing between the two halves is hard,"), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    for(auto p = g.nodes.begin(); p != g.nodes.end(); p++){ p->second.color = TRANSPARENT_BLACK; }
    cs.remove_subscene("ks_bd");

    cs.stage_macroblock(FileBlock("but I think there's something more fundamental going on here."), 1);
    cs.render_microblock();

    for(KlotskiBoard whichboard : {klotski_necklace}) {
        gs->state_manager.set({{"highlight_point_opacity", "0"}});
        for(auto& p : g.nodes){ p.second.color = 0x00000000; }

        cs.stage_macroblock(FileBlock("Let's peek at a faraway land,"), 2);
        double hash = gs->next_hash = whichboard.get_hash();
        cs.render_microblock();
        gs->state_manager.transition(MICRO, {{"d", ".02"}});
        cs.render_microblock();

        gs->state_manager.set({{"points_opacity", "1"}, {"points_radius_multiplier", "0"}});
        gs->state_manager.transition(MICRO, {{"points_radius_multiplier", "1"}});
        Node& n = g.nodes.find(hash)->second;
        unordered_set<double> n_hashes = g.get_neighborhood(hash, 4);
        list<string> strings;
        for(double d : n_hashes){
            Node& n = g.nodes.find(d)->second;
            n.color = 0xffffff00;
            strings.push_back(n.data->representation);
        }
        cs.stage_macroblock(FileBlock("and select a small region of nodes."), 1);
        cs.render_microblock();

        cs.stage_macroblock(FileBlock("Now I'm going to show you the boards for all of those nodes,"), 1);
        int i = 0;
        for(const string& s : strings){
            shared_ptr<KlotskiScene> tksn = make_shared<KlotskiScene>(KlotskiBoard(4, 5, s, false));
            tksn->state_manager.set(board_width_height);
            string key = "tks" + to_string(i);
            cs.add_scene_fade_in(MICRO, tksn, key, -.15, yval, 1.5/strings.size());
            cs.slide_subscene(MICRO, key, .3, 0);
            i++;
        }
        cs.render_microblock();

        cs.stage_macroblock(FileBlock("but blurred together so that we can only see the shared patterns."), 1);
        cs.render_microblock();

        cs.stage_macroblock(FileBlock("All the boards in this substructure have one thing in common- the block and bar are in the center."), 2);
        for(int i = 0; i < strings.size(); i++) {
            string key = "tks" + to_string(i);
            cs.state_manager.transition(MICRO, {{key + ".x",".5"},{key + ".y",".5"}});
        }
        cs.render_microblock();
        cs.render_microblock();

        cs.stage_macroblock(FileBlock("This local structure is formed by two holes, similar to the 2d grid before,"), 1);
        shared_ptr<KlotskiScene> center = make_shared<KlotskiScene>(whichboard);
        center->state_manager.set(board_width_height);
        cs.add_scene_fade_in(MICRO, center, "center");
        for(int i = 0; i < strings.size(); i++) {
            cs.slide_subscene(MICRO, "tks" + to_string(i), -1, 0);
        }
        cs.render_microblock();

        cs.stage_macroblock(SilenceBlock(1), 1);
        cs.state_manager.transition(MICRO, {{"center.x",".15"},{"center.y",to_string(yval)}});
        cs.render_microblock();

        for(int i = 0; i < strings.size(); i++) {
            cs.remove_subscene("tks" + to_string(i));
        }
        gs->state_manager.transition(MACRO, {{"highlight_point_opacity", "1"}});
        perform_shortest_path_with_graph(cs, gs, center, KlotskiBoard(4, 5, "afgcabbcebbheddhi..j", false), FileBlock("but if we align the holes just right,"));
        perform_shortest_path_with_graph(cs, gs, center, klotski_necklace_2, FileBlock("we can break out into a different substructure with its own rules and organization!"));

        cs.state_manager.transition(MICRO, {{"center.x","-.15"}});
        cs.stage_macroblock(SilenceBlock(1), 1);
        n_hashes = g.get_neighborhood(klotski_necklace_2.get_hash(), 6);
        strings.clear();
        for(double d : n_hashes){
            Node& n = g.nodes.find(d)->second;
            n.color = 0xffffff00;
            strings.push_back(n.data->representation);
        }
        int str_i = 0;
        for(const string& s : strings){
            shared_ptr<KlotskiScene> tksn = make_shared<KlotskiScene>(KlotskiBoard(4, 5, s, false));
            tksn->state_manager.set(board_width_height);
            string key = "tks" + to_string(str_i);
            cs.add_scene_fade_in(MICRO, tksn, key, -.15, yval, 1.5/strings.size());
            cs.slide_subscene(MICRO, key, .3, 0);
            str_i++;
        }
        cs.render_microblock();
        cs.remove_subscene("center");

        cs.fade_subscene(MACRO, "center", 0);
        cs.stage_macroblock(SilenceBlock(4), 4);
        gs->state_manager.transition(MACRO, {{"points_opacity", "0"}});
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        for(int i = 0; i < strings.size(); i++) {
            cs.slide_subscene(MICRO, "tks" + to_string(i), -1, 0);
        }
        gs->state_manager.transition(MICRO, {{"d", ".5"}});
        cs.render_microblock();
        for(int i = 0; i < strings.size(); i++) {
            cs.remove_subscene("tks" + to_string(i));
        }
    }

    gs->next_hash = 0;
    gs->state_manager.transition(MACRO, {{"d", ".2"}});
    cs.stage_macroblock(CompositeBlock(FileBlock("In other words, this graph can be thought of not as a single unified topology,"), FileBlock("a single overarching framework of rules which govern the puzzle as a whole,")), 1);
    double hash = gs->next_hash = klotski_earring.get_hash();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("but rather a collection of local sub-puzzles"), FileBlock("which have their own logic and form,")), 1*4);
    for(KlotskiBoard whichboard : {/*klotski_necklace,*/ klotski_earring/*, klotski_whisker*/}) {
        gs->state_manager.set({{"highlight_point_opacity", "0"}});
        for(auto& p : g.nodes){ p.second.color = 0x00000000; }

        gs->state_manager.transition(MICRO, {{"d", ".03"}});
        cs.render_microblock();

        gs->state_manager.set({{"points_opacity", "1"}, {"points_radius_multiplier", "0"}});
        gs->state_manager.transition(MICRO, {{"points_radius_multiplier", "1"}});
        Node& n = g.nodes.find(hash)->second;
        unordered_set<double> n_hashes = g.get_neighborhood(hash, 5);
        list<string> strings;
        for(double d : n_hashes){
            Node& n = g.nodes.find(d)->second;
            n.color = 0xffffff00;
            strings.push_back(n.data->representation);
        }
        cs.render_microblock();

        int i = 0;
        for(const string& s : strings){
            shared_ptr<KlotskiScene> tksn = make_shared<KlotskiScene>(KlotskiBoard(4, 5, s, false));
            tksn->state_manager.set(board_width_height);
            string key = "tks" + to_string(i);
            cs.add_scene_fade_in(MACRO, tksn, key, -.15, yval, 1.5/strings.size());
            i++;
            cs.slide_subscene(MICRO, key, .3, 0);
        }
        cs.render_microblock();

        gs->state_manager.transition(MICRO, {{"points_opacity", "0"}});
        cs.render_microblock();
        for(int i = 0; i < strings.size(); i++) {
            cs.slide_subscene(MACRO, "tks" + to_string(i), -.3, 0);
        }
    }

    cs.stage_macroblock(FileBlock("loosely stitched together into one large puzzle."), 1);
    gs->state_manager.transition(MACRO, {{"d", "1"}});
    gs->next_hash = 0;
    cs.render_microblock();

    cs.slide_subscene(MICRO, "gs", -.25, 0);
    cs.stage_macroblock(FileBlock("And this seems like a trend among graph-based emergent structures...!"), 1);
    cs.render_microblock();

    Graph fg;
    shared_ptr<C4GraphScene> fgs = make_shared<C4GraphScene>(&fg, false, "", TRIM_STEADY_STATES, .5, 1);

    StateSet c4_default_graph_state{
        {"q1", "1"},
        {"qi", "<t> .2 * cos"},
        {"qj", "<t> .314 * sin"},
        {"qk", "0"}, // Camera orientation quaternion
        {"decay",".6"},
        {"dimensions","3.98"},
        {"surfaces_opacity","0"},
        {"points_opacity","0"},
        {"physics_multiplier","100"}, // How many times to iterate the graph-spreader
    };
    fgs->state_manager.set(c4_default_graph_state);
    cs.add_scene_fade_in(MICRO, fgs, "fgs", .75, .5);
    cs.stage_macroblock(FileBlock("Stay tuned to see how strategy board game solutions take that same form."), 2);
    cs.render_microblock();
    cs.render_microblock();

}

void part9_old(){
    // Recursively find all of the possible states including those which are unreachable
    Graph omni;
    unordered_set<string> set;
    recursive_placer(set, "....................", 0);
    cout << "Set size: " + to_string(set.size()) << endl;
    for(const string& s : set) {
        omni.add_to_stack(new KlotskiBoard(4, 5, s, false));
    }

    while(omni.nodes.size() > 0) {
        cout << "REMAINING NODES: " << omni.nodes.size() << endl;
        Graph g;
        auto it = omni.nodes.begin();

        KlotskiBoard* kb = new KlotskiBoard(4, 5, it->second.data->representation, false); 
        KlotskiBoard* m0 = new KlotskiBoard(4, 5, omni.nodes.find(kb->get_reverse_hash())->second.data->representation, false);
        KlotskiBoard* m1 = new KlotskiBoard(4, 5, omni.nodes.find(kb->get_reverse_hash_2())->second.data->representation, false);
        KlotskiBoard* m2 = new KlotskiBoard(4, 5, omni.nodes.find(m0->get_reverse_hash_2())->second.data->representation, false);
        g.add_to_stack(kb);
        g.add_to_stack(m0);
        g.add_to_stack(m1);
        g.add_to_stack(m2);

        g.expand(-1);
        int sz = g.size();

        if(sz > 150 && sz < 20000){
            CompositeScene cs;
            shared_ptr<KlotskiScene> ks = make_shared<KlotskiScene>(KlotskiBoard(*kb));
            shared_ptr<GraphScene> gs = make_shared<GraphScene>(&g, false);
            gs->state_manager.set({{"mirror_force", ".1"}, {"physics_multiplier", "10"}, {"points_opacity", "0"}, {"dimensions", "3.99"}});
            shared_ptr<LatexScene> ls = make_shared<LatexScene>("Node Count: " + to_string(g.size()), 1, .3, .3);
            cs.add_scene(ks, "ks");
            cs.add_scene(gs, "gs");
            cs.state_manager.set(board_position);
            ks->state_manager.set(board_width_height);
            cs.add_scene(ls, "ls", .15, .9);
            cs.stage_macroblock(SilenceBlock(1), 1);
            cs.render_microblock();
        }

        for(auto& p : g.nodes) omni.remove_node(p.first);
    }
}

void promo1(){
    {
        CompositeScene cs;
        shared_ptr<Mp4Scene> ms = make_shared<Mp4Scene>(vector<string>{"250423_Thinking in Code_Sequence", "Sequence_Calc_250616"});
        cs.stage_macroblock(CompositeBlock(FileBlock("I love using Computer Science to shine a fresh light on systems we otherwise wouldn't think twice about."), FileBlock("Complicated topics- slidy puzzles and programming alike, when viewed from the right perspective, can become much more clear.")), 5);
        cs.add_scene(ms, "ms");
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        cs.fade_all_subscenes(MICRO, 0);
        cs.render_microblock();
    }

    Mp4Scene ms = Mp4Scene(vector<string>{"Logo Animation_mp4", "CC_TheForLoop_cropped_v01", "Content Footage_How AI Works_Digit Checker", "Logic_Sequence_250616", "Sequence_HowAIworks_250616", "Sequence_Circuits_2506"});
    ms.stage_macroblock(CompositeBlock(CompositeBlock(FileBlock("That's why Brilliant's lessons are so effective."), FileBlock("Their games and clear examples make all those hard subjects infinitely more digestible.")), CompositeBlock(FileBlock("Learn to think like a programmer by breaking down complex problems into manageable chunks of code, or dive right into Python and start building programs on day one."), CompositeBlock(FileBlock("Brilliant helps you get smarter every day, with thousands of interactive lessons in math, science, programming, data analysis, and AI."), FileBlock("These topics don't have to be intimidating- uncover their mystery with Brilliant!")))), 1);
    ms.render_microblock();

    {
        CompositeScene cs;
        shared_ptr<Mp4Scene> csms = make_shared<Mp4Scene>("main");
        cs.add_scene(csms, "csms");
        shared_ptr<PngScene> ps = make_shared<PngScene>("QR Code - 2swap", .4, .4);
        cs.add_scene(ps, "ps", -.25, .7);
        cs.stage_macroblock(FileBlock("To try everything Brilliant has to offer for free for 30 days and get 20% off an annual subscription, visit brilliant.org/2swap in the description, or scan the QR code onscreen."), 5);
        cs.render_microblock();
        cs.slide_subscene(MICRO, "ps", .42, 0);
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
    }
}

void outtro() {
    CompositeScene cs;
    cs.stage_macroblock(FileBlock("This has been 2swap."), 1);
    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<TwoswapScene> tss = make_shared<TwoswapScene>();
    shared_ptr<PngScene> note = make_shared<PngScene>("note", 0.18, 0.18);
    shared_ptr<LatexScene> seef = make_shared<LatexScene>(latex_text("6884"), 1, .6, .27);
    cs.add_scene_fade_in(MICRO, tss, "tss");
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("with music by 6884"), 4);
    cs.add_scene_fade_in(MICRO, seef, "seef", 0.6, 0.73);
    cs.add_scene_fade_in(MICRO, note, "note", 0.44, 0.73);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
}

void render_video() {
    Graph* tri = new Graph;
    shared_ptr<GraphScene> tgs;
    shared_ptr<KlotskiScene> tks;

    part0();
    part1();
    part2();
    part3(tri, tks, tgs);
    part4();
    part5();
    part6();
    part7(tgs, tks);
    promo1();
    outtro();

    delete tri;
}
