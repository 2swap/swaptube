using namespace std;
void beginning() {
    PRINT_TO_TERMINAL = false;
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.4;
    g.gravity_strength = 0;
    g.dimensions = 2;
    g.sqrty = true;
    g.lock_root_at_origin = true;
    C4GraphScene c4(&g, "444", MANUAL);
    c4.physics_multiplier = 1;

    std::unordered_map<std::string, std::string> closequat{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "4"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
    };
    auto closequat_moving = closequat;
    closequat_moving["q1"] = "<t> 4 / cos";
    closequat_moving["qj"] = "<t> -4 / sin";
    dag.add_equations(closequat_moving);
    c4.inject_audio_and_render(AudioSegment("So far we have been interested in strategies of connect 4,"));
    dag.add_transition("d", "10");
    c4.inject_audio_and_render(AudioSegment("but now it's time to apply those strategies to actually get a structural understanding of the game."));

    c4.inject_audio(AudioSegment(2), 7+49);
    if(FOR_REAL){
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board("444" + to_string(i)));
        c4.render();
    }
    g.dimensions = 3;
    for(int i = 1; i <= 7; i++){
        for(int j = 1; j <= 7; j++){
            g.add_node(new C4Board("444" + to_string(i) + to_string(j)));
            c4.render();
        }
    }
    }
    dag.add_transition("surfaces_opacity", "0");
    c4.inject_audio_and_render(AudioSegment("First of all, you're gonna have to get comfortable imagining the game as a tree."));

    dag.add_transition("y", "30");
    c4.inject_audio_and_render(AudioSegment(1));
    g.clear();


    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"y", "1.5"}, {"d", "6"},
        {"q1", "0"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"lines_opacity", "0"},
    });
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"}
    });

    g.add_node(new C4Board("436"));
    
    c4.inject_audio_and_render(AudioSegment("For any starting position, we can draw a node."));
    g.gravity_strength = 1;
    c4.physics_multiplier = 0;
    g.dimensions = 2;
    c4.inject_audio_and_render(AudioSegment("there is some amount of moves that can be made from that position, which we will represent as nodes connected to the root."));

    double x = -1.5*4;
    double y = 2;
    g.add_node_with_position(new C4Board("4361"), (x+=1.5), y, 0);
    c4.inject_audio_and_render(AudioSegment("Yellow can play in the left column,"));

    g.add_node_with_position(new C4Board("4362"), (x+=1.5), y, 0);
    c4.inject_audio_and_render(AudioSegment("the next column over,"));

    c4.inject_audio(AudioSegment("and so on successively."), 5);
    for(int i = 3; i <= 7; i++){
        g.add_node_with_position(new C4Board("436" + to_string(i)), (x+=1.5), y, 0);
        c4.render();
    }
    g.immobilize_all_nodes();

    c4.physics_multiplier = 1;
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"lines_opacity", "1"}
    });
    c4.inject_audio_and_render(AudioSegment("We connect those new nodes to the root with yellow lines, since yellow was the one who played a move here."));
    c4.inject_audio(AudioSegment("And from those positions, there are even more options that can be made, by Red this time, continuing the graph."), 49);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "15"},
        {"y", "3"},
        {"surfaces_opacity", "0"},
    });
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++){
        g.add_node_with_position(new C4Board("436" + to_string(i) + to_string(j)), (i-4)*1.5, 4, 0);
        c4.render();
    }







    c4.inject_audio(AudioSegment("In other words, every path through this graph represents a particular continuation from the existing board."), 4);
    for(auto& node : g.nodes)
        for (auto& edge : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors))
            const_cast<Edge&>(edge).opacity = 0.2;
    c4.render();
    for(auto& node : g.nodes)
        for (auto& edge : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors)){
            bool highlight = false;
            highlight |= g.nodes.find(edge.to)->second.data->representation == "43667" && g.nodes.find(edge.from)->second.data->representation == "4366";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "4366" && g.nodes.find(edge.from)->second.data->representation == "43667";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "4366" && g.nodes.find(edge.from)->second.data->representation == "436";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "436" && g.nodes.find(edge.from)->second.data->representation == "4366";
            const_cast<Edge&>(edge).opacity = highlight ? 1 : 0.2;
        }
    c4.render();
    for(auto& node : g.nodes)
        for (auto& edge : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors)){
            bool highlight = false;
            highlight |= g.nodes.find(edge.to)->second.data->representation == "43626" && g.nodes.find(edge.from)->second.data->representation == "4362";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "4362" && g.nodes.find(edge.from)->second.data->representation == "43626";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "4362" && g.nodes.find(edge.from)->second.data->representation == "436";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "436" && g.nodes.find(edge.from)->second.data->representation == "4362";
            const_cast<Edge&>(edge).opacity = highlight ? 1 : 0.2;
        }
    c4.render();
    for(auto& node : g.nodes)
        for (auto& edge : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors)){
            bool highlight = false;
            highlight |= g.nodes.find(edge.to)->second.data->representation == "43655" && g.nodes.find(edge.from)->second.data->representation == "4365";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "4365" && g.nodes.find(edge.from)->second.data->representation == "43655";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "4365" && g.nodes.find(edge.from)->second.data->representation == "436";
            highlight |= g.nodes.find(edge.to)->second.data->representation == "436" && g.nodes.find(edge.from)->second.data->representation == "4365";
            const_cast<Edge&>(edge).opacity = highlight ? 1 : 0.2;
        }
    c4.render();
    for(auto& node : g.nodes)
        for (auto& edge : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors))
            const_cast<Edge&>(edge).opacity = 1;
    c4.inject_audio_and_render(AudioSegment("Any set of moves that you could make is in this graph."));

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", "50"},
    });
    c4.inject_audio_and_render(AudioSegment("As a result, it gets intractably large, really fast."));
    g.dimensions = 3;
    g.gravity_strength = 1;
    g.clear();
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"d", "5"},
        {"points_opacity", "1"},
        {"surfaces_opacity", "0"},
        {"q1", "<t> 4 / cos"},
        {"qj", "<t> -4 / sin"},
    });

    C4GraphScene c4gs(VIDEO_WIDTH/2, VIDEO_HEIGHT, &g, "", MANUAL);
    LatexScene latex(VIDEO_WIDTH/2, VIDEO_HEIGHT, "0: 1");
    CompositeScene composite;
    composite.add_scene(&c4gs, 0, 0, .5, 1);
    composite.add_scene(&latex, .5, 0, .5, 1);

    composite.inject_audio_and_render(AudioSegment("At any position in the opening, there are 7 possible moves to make. This means, at a depth of 0, the amount of nodes is 1 (the empty board),"));

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "10"},
        {"y", "5"},
    });
    composite.inject_audio(AudioSegment("at a depth of 1, the amount of nodes is 7 (the 7 openings),"), 7);
    latex.append_transition("\\\\\\\\1: 7");
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board(to_string(i)));
        composite.render();
    }
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "0"},
    });

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "20"},
        {"y", "10"},
    });
    composite.inject_audio(AudioSegment("at a depth of 2, the amount of nodes is 49,"), 49);
    latex.append_transition("\\\\\\\\2: 49");
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++){
        g.add_node(new C4Board(to_string(i) + to_string(j)));
        composite.render();
    }

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "50"},
        {"y", "20"},
    });
    composite.inject_audio(AudioSegment("and the progression continues exponentially."), 343);
    latex.append_transition("\\\\\\\\3: 238");
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 7; k++){
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k)));
        composite.render();
    }

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "70"},
        {"y", "30"},
    });
    composite.inject_audio(AudioSegment(3), 2401);
    latex.append_transition("\\\\\\\\4: 1120");
    if(FOR_REAL)for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 7; k++)for(int l = 1; l <= 7; l++){
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k) + to_string(l)));
        composite.render();
    }
    composite.inject_audio_and_render(AudioSegment("Now this is just a mess. You can't tell the structure of this graph whatsoever by looking at it, and we're only 4 moves in."));
    latex.append_transition("\\\\\\\\...\\\\\\\\total: 4,531,985,219,092");
    composite.inject_audio_and_render(AudioSegment("In fact, Stefan Edelkamp and Peter Kissmann computed in 2008 that there are a total of 4.5 trillion unique positions at any depth."));

    C4Scene c4_left(VIDEO_WIDTH*2/5, VIDEO_HEIGHT, "43637563");
    LatexScene latex_equals(2*VIDEO_WIDTH/5, VIDEO_HEIGHT, "=");
    C4Scene c4_right(VIDEO_WIDTH*2/5, VIDEO_HEIGHT, "45251325");
    CompositeScene mirror_symmetry_composite;
    mirror_symmetry_composite.add_scene(&latex_equals, .3, 0, .4, 1);
    mirror_symmetry_composite.add_scene(&c4_left, 0, 0, .4, 1);
    mirror_symmetry_composite.add_scene(&c4_right, .6, 0, .4, 1);
    mirror_symmetry_composite.inject_audio_and_render(AudioSegment("We can do some tricks like de-duplicating based on horizontal symmetry,"));
    LatexScene latex_lower_bound(VIDEO_WIDTH, VIDEO_HEIGHT/5, "Best Case: 2,265,992,609,546");
    mirror_symmetry_composite.add_scene(&latex_lower_bound, 0, .8, 1, .2);
    mirror_symmetry_composite.inject_audio_and_render(AudioSegment("but that will remove less than half of the nodes, because we are only deleting one node per mirror pair."));

    C4GraphScene just_the_graph(VIDEO_WIDTH, VIDEO_HEIGHT, &g, "", MANUAL);

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "0"},
        {"lines_opacity", "0"},
        {"points_opacity", "0"},
    });
    just_the_graph.inject_audio_and_render(AudioSegment("We are going to need a bit of a paradigm shift if we want to be able to gain any insight from this tangled mess."));
    g.clear();
    g.add_node(new C4Board(""));

    dag.add_equations(closequat);
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
        {"d", "20"},
        {"y", "10"},
    });
    just_the_graph.inject_audio(AudioSegment("Instead of growing the graph from the beginning of the game out,"), 7+49);
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board(to_string(i)));
        just_the_graph.render();
    }
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++){
        g.add_node(new C4Board(to_string(i) + to_string(j)));
        just_the_graph.render();
    }
    just_the_graph.inject_audio_and_render(AudioSegment("Let's start with an endgame position to make life a little bit easier on ourselves."));

}
void endgame_examination(){
    std::unordered_map<std::string, std::string> closequat{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "4"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
    };
    auto closequat_moving = closequat;
    closequat_moving["q1"] = "<t> 8 / cos";
    closequat_moving["qj"] = "<t> -8 / sin";
    dag.add_equations(closequat_moving);

    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"d", "3"},
        {"y", "0"},
        {"surfaces_opacity", "1"},
    });
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.3;
    g.gravity_strength = 0;
    g.dimensions = 3;
    g.sqrty = true;
    g.lock_root_at_origin = true;
    C4GraphScene c4(&g, "36426444226456412121132335635611737", FULL);
    c4.physics_multiplier = 1;
    c4.inject_audio_and_render(AudioSegment("This is a particular endgame."));
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "0"},
        {"points_opacity", "0.1"},
        {"d", "30"},
    });
    c4.inject_audio_and_render(AudioSegment("Let's see what it looks like when we expand out its full tree of positions."));
    c4.inject_audio(AudioSegment(6), 100);
    for(int i = 0; i < 99; i++){
        g.expand_graph(false, true);
        c4.render();
    }
    g.expand_graph(false, false);
    c4.render();
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"lines_opacity", "0.1"},
    });
    c4.color_edges = false;
    c4.inject_audio_and_render(AudioSegment("Now, I want to focus on the nodes of this graph."));
    for(auto& node : g.nodes)
        if(node.second.data->who_won() == INCOMPLETE)
            node.second.opacity = 0;
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"points_opacity", "1"},
    });
    c4.inject_audio_and_render(AudioSegment("Let's identify all the terminal states of this board- that is, the nodes which have no children."));
    double some_red_node = -1;
    for(auto& node : g.nodes)
        if(node.second.data->who_won() == RED){
            node.second.color = C4_RED;
            some_red_node = node.second.hash;
        }
    c4.inject_audio_and_render(AudioSegment("In other words, we are identifying the games which have either been won for red,"));
    double some_yellow_node = -1;
    for(auto& node : g.nodes)
        if(node.second.data->who_won() == YELLOW){
            node.second.color = 0xffaaaa00;
            some_yellow_node = node.second.hash;
        }
    c4.inject_audio_and_render(AudioSegment("won for yellow,"));
    double some_tie_node = -1;
    for(auto& node : g.nodes)
        if(node.second.data->who_won() == TIE){
            node.second.color = 0xff5555ff;
            some_tie_node = node.second.hash;
        }
    c4.inject_audio_and_render(AudioSegment("or have completely filled without a winner as a tie."));

    c4.inject_audio_and_render(AudioSegment("Now, it's easier to tell what's going on."));

    auto result = g.shortest_path(g.root_node_hash, some_red_node);
    list<Edge*> edges = result.second;
    for(Edge* e : edges) e->opacity = 5;
    c4.inject_audio_and_render(AudioSegment("In the case of this endgame, there are some paths that lead to a red victory,"));
    for(Edge* e : edges) e->opacity = 1;

    result = g.shortest_path(g.root_node_hash, some_yellow_node);
    edges = result.second;
    for(Edge* e : edges) e->opacity = 5;
    c4.inject_audio_and_render(AudioSegment("some paths that lead to a Yellow victory,"));
    for(Edge* e : edges) e->opacity = 1;

    result = g.shortest_path(g.root_node_hash, some_tie_node);
    edges = result.second;
    for(Edge* e : edges) e->opacity = 5;
    c4.inject_audio_and_render(AudioSegment("and some paths that lead to the game becoming a tie."));
    for(Edge* e : edges) e->opacity = 1;


    FOR_REAL = true;
    C4Scene board(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, "36426444226456412121132335635611737");
    CompositeScene composite;
    composite.add_scene(&c4, 0, 0, 1, 1);
    composite.add_scene(&board, 0, 0, .333, .333);

    composite.inject_audio_and_render(AudioSegment("So, what we might want to know is, who is favored in the root position? If both players play perfectly, who wins?"));
    composite.inject_audio_and_render(AudioSegment("We have constructed this graph, so how can we use it to play optimally in this position?"));
    g.immobilize_all_nodes();
    composite.inject_audio_and_render(AudioSegment("Hold that thought- we'll get there. We'll have to work our way up to that point."));

    Node<C4Board>* consider;
    for(auto& node : g.nodes){
        if(node.second.color != 0xffffffff) continue;
        if(node.second.neighbors.size() != 1) continue;
        bool bad = false;
        for(const Edge& e : node.second.neighbors) if(g.nodes.at(e.to).color == 0xffffffff) bad = true;
        if(bad) continue;
        consider = &(node.second);
        break;
    }
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
        {"d", "10"},
    });
    composite.inject_audio_and_render(AudioSegment("Consider this board, which is in the graph. The game hasn't ended yet, but it is about to."));
    composite.inject_audio_and_render(AudioSegment("It's Red's move, with only one choice, which is to win the game."));
    int dummy;
    C4Result winner = consider->data->who_is_winning(dummy);
    consider->color = winner==RED?C4_RED:(winner==YELLOW?0xffaaaa00:0xff5555ff);
    consider->opacity = 1;
    consider->highlight = false;
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "30"},
    });
    composite.inject_audio_and_render(AudioSegment("So, for all intents and purposes, we may as well color this node red too. Red will win here."));

    composite.inject_audio(AudioSegment("We can repeat this line of thinking for all of the other moves which force the game into an ending state."), 30);
    while(true){
        consider = NULL;
        for(auto& node : g.nodes){
            if(node.second.color != 0xffffffff) continue;
            if(node.second.neighbors.size() != 1) continue;
            bool bad = false;
            for(const Edge& e : node.second.neighbors) if(g.nodes.at(e.to).color == 0xffffffff) bad = true;
            if(bad) continue;
            consider = &(node.second);
            break;
        }
        if(consider == NULL) break;
        consider->highlight = true;
        board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
        composite.render();
        C4Result winner = consider->data->who_is_winning(dummy);
        consider->color = winner==RED?C4_RED:(winner==YELLOW?0xffaaaa00:0xff5555ff);
        consider->opacity = 1;
        consider->highlight = false;
        composite.render();
    }
    composite.inject_audio_and_render(AudioSegment("Awesome! But, now, we are out of obvious nodes to fill in."));
    for(auto& node : g.nodes){
        if(node.second.color != 0xffffffff) continue;
        bool bad = false;
        for(const Edge& e : node.second.neighbors) if(g.nodes.at(e.to).color == 0xffffffff) bad = true;
        if(bad) continue;
        consider = &(node.second);
        break;
    }
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
        {"d", "10"},
    });
    composite.inject_audio_and_render(AudioSegment("What about this one?"));
    
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("There's not only one choice, but both of the children have already been colored with a computed game result."));
    c4.color_edges = true;
    composite.inject_audio_and_render(AudioSegment("If we color the edges again, we recall that it's yellow to move in this position."));
    composite.inject_audio_and_render(AudioSegment("Notice that we have already computed the two branches' results- one is yellow's win, and one is a tie."));
    composite.inject_audio_and_render(AudioSegment("Well, Yellow will naturally take the win over a tie."));
    winner = consider->data->who_is_winning(dummy);
    consider->color = winner==RED?C4_RED:(winner==YELLOW?0xffaaaa00:0xff5555ff);
    consider->opacity = 1;
    consider->highlight = false;
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "30"},
    });
    composite.inject_audio_and_render(AudioSegment("Therefore, we can color this node as yellow."));

    composite.inject_audio(AudioSegment("In general, a node's status is the best result of all of the children, on behalf of the player whose move it is."), 32);
    while(true){
        consider = NULL;
        for(auto& node : g.nodes){
            if(node.second.color != 0xffffffff) continue;
            bool bad = false;
            for(const Edge& e : node.second.neighbors) if(g.nodes.at(e.to).color == 0xffffffff) bad = true;
            if(bad) continue;
            consider = &(node.second);
            break;
        }
        if(consider == NULL) break;
        for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
        consider->highlight = true;
        board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
        composite.render();
        for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
        winner = consider->data->who_is_winning(dummy);
        consider->color = winner==RED?C4_RED:(winner==YELLOW?0xffaaaa00:0xff5555ff);
        consider->opacity = 1;
        consider->highlight = false;
        composite.render();
    }

    g.nodes.at(g.root_node_hash).highlight = true;

    composite.inject_audio_and_render(AudioSegment("And now, we have worked our way all the way back up to the endgame which we considered in the first place, and it turns out that it was a tie!"));

}

void render_video() {
    FOR_REAL = false;
    if(FOR_REAL) beginning();
    endgame_examination();


    /*
    v.inject_audio_and_render(AudioSegment("To know the strategy that Red must employ to win, we simply move along the graph by only selecting nodes which are colored red, to guarantee we arrive at a red win."));
    v.inject_audio_and_render(AudioSegment("We can change our coloring scheme to show who is winning on a certain path."));
    v.inject_audio_and_render(AudioSegment("This algorithm for determining who is winning is called Minimax."));
    v.inject_audio_and_render(AudioSegment("So the immediate question then becomes, can we apply it to the opening to determine who should win for the empty board?"));
    v.inject_audio_and_render(AudioSegment("In theory, sure, but even with our modern computers, we can't do this for 4 trillion nodes. It's just too much."));
    v.inject_audio_and_render(AudioSegment("But there are a number of clever tricks that you can do."));
    v.inject_audio_and_render(AudioSegment("If we already determined that, on a Red move, there is a single Red child, then there is no need to search the remaining variations. We can simply discard them."));
    v.inject_audio_and_render(AudioSegment("This turns out to be really big, because at all resolutions of the exponential growth of the game tree, you are shaving off entire enormous branches of the game tree which don't need to be evaluated."));
    v.inject_audio_and_render(AudioSegment("For a specific example, if we are here, we already know that Red can just win by making 4 in a row. There is no point in searching any other branches than this one. It would be a waste of our time."));
    v.inject_audio_and_render(AudioSegment("One last trick that we can take advantage of, is we can guide our search heuristically. If we can make good guesses programmatically about which branches look strong, then this branch-pruning method becomes even more potent."));
    v.inject_audio_and_render(AudioSegment("Using these strategies, Connect 4 was solved. Thanks to computer scientists such as James Dow Allen, Victor Allis, and John Tromp, we know that Red Wins, and only does so by starting with a disk in the center column."));
    v.inject_audio_and_render(AudioSegment("One space off to either side and it's a tie, and starting on the edge or one in from the edge, it's Yellow to win."));
    v.inject_audio_and_render(AudioSegment("But why?"));
    v.inject_audio_and_render(AudioSegment("This is great and all, but it provides absolutely no information into why, or how, this is the case."));
    v.inject_audio_and_render(AudioSegment("It was computed by means of a brute force search, and contributes little to the theoretical analysis of the game."));
    v.inject_audio_and_render(AudioSegment("All of the strategy, and actual understanding of the game, cannot be communicated by an immense tree of colored nodes. "));
    v.inject_audio_and_render(AudioSegment("Is there any way to rigorously unite the wisdom and intuition accrued by countless human experts with the output of this enormous algorithm?"));
    v.inject_audio_and_render(AudioSegment("We can at least do better."));
    v.inject_audio_and_render(AudioSegment("Before we can truly demystify Connect 4, I need to explain different types of solutions."));
    v.inject_audio_and_render(AudioSegment("What we talked about before, where I labeled every node in the graph with a color indicating who wins- that is an example of a strong solution."));
    v.inject_audio_and_render(AudioSegment("We know, for ANY starting position, who is going to win the game."));
    v.inject_audio_and_render(AudioSegment("But we also mentioned weak solutions. A weak solution is where you know _just enough_ of the game graph to be able to guarantee a win."));
    v.inject_audio_and_render(AudioSegment("When we were cutting off branches that didn't need to be searched, that was an example of a weak solution. We don't need to memorize stupid variants that we would never even get ourselves into, would we?"));
    v.inject_audio_and_render(AudioSegment("A strong solution tells us everything that could be known, but weak solutions have a lot more room for creative expression."));
    v.inject_audio_and_render(AudioSegment("Can we immensely reduce the amount of information required to 'know' a weak solution?"));
    v.inject_audio_and_render(AudioSegment("Is there a small subset of opening lines that you need to know?"));
    v.inject_audio_and_render(AudioSegment("Are there any particularly clever ways of representing those weak solutions? We can represent them graph-theoretically, but perhaps they can be expressed by other means?"));
    v.inject_audio_and_render(AudioSegment("These questions are hard to answer."));
    v.inject_audio_and_render(AudioSegment("Perhaps that's because they require an intersectional understanding of computer science and a technical understanding of emergent complexities of connect 4 play."));
    v.inject_audio_and_render(AudioSegment("As far as I can tell, no one has seriously taken up the task of answering them... until now!"));
    v.inject_audio_and_render(AudioSegment("The answer to all of these questions turns out to be an astonishing yes."));
    v.inject_audio_and_render(AudioSegment("There ARE extremely low information-density weak solutions with clever representations using Claimeven which we talked about last time."));
    v.inject_audio_and_render(AudioSegment("We CAN collapse the opening down to a reduced set of irreducibly complex openings."));
    v.inject_audio_and_render(AudioSegment("There IS a notation by which we can concisely convey the weak solution of sufficiently developed positions."));
    v.inject_audio_and_render(AudioSegment("What's the point of all this? Through enormous amounts of compute, we were capable of taming connect 4 computationally."));
    v.inject_audio_and_render(AudioSegment("But the solution that was found is a completely abstract object."));
    v.inject_audio_and_render(AudioSegment("You can't realistically visualize it. You can't realistically memorize it."));
    v.inject_audio_and_render(AudioSegment("It may be a landmark in computer science to be able to develop the tools to perform such a task, and I don't mean to take away from the importance of _this result_ (pointing at the opening board)."));
    v.inject_audio_and_render(AudioSegment("But, the search that was performed yielded a completely intangible theoretical object which we can only get a faint sense for by querying for who wins, one by one, position by position."));
    v.inject_audio_and_render(AudioSegment("But what is it's form? What is its structure?"));
    v.inject_audio_and_render(AudioSegment("My goal is to throw all the tricks in the book at this monster and show you what it looks like up close."));
    v.inject_audio_and_render(AudioSegment("Now, I may be biased, but I think the next few videos are going to be _really_ cool. We're going to take connect four as a case study in the nature of emergent complexity itself. Stay tuned. This has been 2swap."));

    
    */
}