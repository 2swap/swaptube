using namespace std;
void beginning() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.4;
    g.dimensions = 2;
    C4GraphScene c4(&g, "444", MANUAL);

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
    c4.inject_audio_and_render(AudioSegment("So far we've been interested in strategies of connect 4,"));
    dag.add_transition("d", "10");
    c4.inject_audio_and_render(AudioSegment("but now it's time to apply those strategies to get a structural understanding of the game."));

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
    
    c4.inject_audio_and_render(AudioSegment("For a position of our choice, we draw a node."));
    g.gravity_strength = 1;
    c4.physics_multiplier = 0;
    g.dimensions = 2;
    c4.inject_audio_and_render(AudioSegment("there are some moves that can be made from that position, which we'll represent as connected child nodes."));

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
    c4.inject_audio_and_render(AudioSegment("As a result, it gets intractably large really quickly."));
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
    LatexScene latex(VIDEO_WIDTH/2, VIDEO_HEIGHT, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline \\end{tabular}", 0.5);
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline \\end{tabular}");
    CompositeScene composite;
    composite.add_scene(&c4gs, 0, 0, .5, 1);
    composite.add_scene(&latex, .5, 0, .5, 1);

    composite.inject_audio_and_render(AudioSegment("At any position in the opening, there are 7 possible moves to make. So, at a depth of 0, the amount of nodes is 1 (that is, the empty board),"));

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "10"},
        {"y", "5"},
    });
    composite.inject_audio(AudioSegment("at a depth of 1, the amount of nodes is 7 (the 7 openings),"), 7);
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline \\end{tabular}");
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
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline\\end{tabular}");
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++){
        g.add_node(new C4Board(to_string(i) + to_string(j)));
        composite.render();
    }
    g.gravity_strength = 0;

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "50"},
        {"y", "0"},
    });
    composite.inject_audio(AudioSegment("and the progression continues exponentially."), 343);
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline \\end{tabular}");
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 7; k++){
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k)));
        composite.render();
    }

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "80"},
    });
    composite.inject_audio(AudioSegment(3), 2401);
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline \\end{tabular}");
    if(FOR_REAL)for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 7; k++)for(int l = 1; l <= 7; l++){
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k) + to_string(l)));
        composite.render();
    }
    composite.inject_audio_and_render(AudioSegment("Now this is just a mess. You can't tell the structure of this graph whatsoever, and we're only 4 moves in."));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline ... & ... \\\\\\\\ \\hline \\text{Total} & 4,531,985,219,092 \\\\\\\\ \\hline \\end{tabular}");
    composite.inject_audio_and_render(AudioSegment("In fact, Stefan Edelkamp and Peter Kissmann computed in 2008 that there are a total of 4.5 trillion unique positions at any depth."));

    C4Scene c4_left(VIDEO_WIDTH*2/5, VIDEO_HEIGHT, "43637563");
    LatexScene latex_equals(2*VIDEO_WIDTH/5, VIDEO_HEIGHT, "=", 1);
    C4Scene c4_right(VIDEO_WIDTH*2/5, VIDEO_HEIGHT, "45251325");
    CompositeScene mirror_symmetry_composite;
    mirror_symmetry_composite.add_scene(&latex_equals, .3, 0, .4, 1);
    mirror_symmetry_composite.add_scene(&c4_left, 0, 0, .4, 1);
    mirror_symmetry_composite.add_scene(&c4_right, .6, 0, .4, 1);
    mirror_symmetry_composite.inject_audio_and_render(AudioSegment("We can do some tricks like de-duplicating based on horizontal symmetry,"));
    LatexScene latex_lower_bound(VIDEO_WIDTH, VIDEO_HEIGHT/5, "\\text{Nodes}: 2,265,992,609,546", .75);
    mirror_symmetry_composite.add_scene(&latex_lower_bound, 0, .8, 1, .2);
    mirror_symmetry_composite.inject_audio_and_render(AudioSegment("but that'll only remove less than half of the nodes, because we're only deleting one node per mirror pair."));

    C4GraphScene just_the_graph(VIDEO_WIDTH, VIDEO_HEIGHT, &g, "", MANUAL);

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "0"},
        {"lines_opacity", "0"},
        {"points_opacity", "0"},
    });
    just_the_graph.inject_audio_and_render(AudioSegment("We're gonna need a bit of a paradigm shift if we wanna gain any insight from this tangled mess."));
    g.clear();
    g.gravity_strength = .5;
    g.add_node(new C4Board(""));

    dag.add_equations(closequat);
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
        {"d", "15"},
        {"y", "5"},
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
    just_the_graph.inject_audio_and_render(AudioSegment("Let's start with an endgame position to make life a little easier on ourselves."));

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
    closequat_moving["q1"] = "<t> 13 / cos";
    closequat_moving["qj"] = "<t> -13 / sin";
    dag.add_equations(closequat_moving);

    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"d", "3"},
        {"y", "0"},
        {"surfaces_opacity", "1"},
    });
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.3;
    g.dimensions = 3;
    C4GraphScene c4(&g, "36426444226456412121132335635611737", MANUAL);
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
        {"lines_opacity", "0.2"},
    });
    c4.inject_audio_and_render(AudioSegment("Now, I wanna focus on the nodes of this graph."));
    c4.color_edges = false;
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
    c4.inject_audio_and_render(AudioSegment("In other words, we're identifying the games which either Red won,"));
    double some_yellow_node = -1;
    for(auto& node : g.nodes)
        if(node.second.data->who_won() == YELLOW){
            node.second.color = 0xffaaaa00;
            some_yellow_node = node.second.hash;
        }
    c4.inject_audio_and_render(AudioSegment("Yellow won,"));
    double some_tie_node = -1;
    for(auto& node : g.nodes)
        if(node.second.data->who_won() == TIE){
            node.second.color = 0xff5555ff;
            some_tie_node = node.second.hash;
        }
    c4.inject_audio_and_render(AudioSegment("or completely filled without a winner, making it a tie."));

    g.nodes.at(g.root_node_hash).highlight = true;
    auto result = g.shortest_path(g.root_node_hash, some_red_node);
    list<Edge*> edges = result.second;
    for(Edge* e : edges) e->opacity = 5;
    c4.inject_audio_and_render(AudioSegment("In the case of this endgame, there are some _paths_ that lead to a red victory,"));
    for(Edge* e : edges) e->opacity = 1;

    result = g.shortest_path(g.root_node_hash, some_yellow_node);
    edges = result.second;
    for(Edge* e : edges) e->opacity = 5;
    c4.inject_audio_and_render(AudioSegment("some _paths_ that lead to a Yellow victory,"));
    for(Edge* e : edges) e->opacity = 1;

    result = g.shortest_path(g.root_node_hash, some_tie_node);
    edges = result.second;
    for(Edge* e : edges) e->opacity = 5;
    c4.inject_audio_and_render(AudioSegment("and some _paths_ that lead to the game becoming a tie."));
    for(Edge* e : edges) e->opacity = 1;

    C4Scene board(VIDEO_WIDTH*.4, VIDEO_HEIGHT*.4, "36426444226456412121132335635611737");
    CompositeScene composite;
    composite.add_scene(&c4, 0, 0, 1, 1);
    composite.add_scene(&board, 0, 0, .4, .4);

    composite.inject_audio_and_render(AudioSegment("So, what we might wanna know is, who's favored in the root position? If both players play perfectly, who wins?"));
    composite.inject_audio_and_render(AudioSegment("We've constructed this graph, so how can we use it to play optimally from this position?"));
    if(!FOR_REAL) g.iterate_physics(1000);
    g.immobilize_all_nodes();
    composite.inject_audio_and_render(AudioSegment("We'll have to work our way up to that point."));

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
        {"d", "7"},
    });
    composite.inject_audio_and_render(AudioSegment("Consider this board, which is in the graph. The game hasn't ended yet, but it's about to."));
    composite.inject_audio_and_render(AudioSegment("It's Red's move, with only one choice, which is to win the game."));
    int dummy;
    C4Result winner = consider->data->who_is_winning(dummy);
    consider->color = winner==RED?C4_RED:(winner==YELLOW?0xffaaaa00:0xff5555ff);
    consider->opacity = 1;
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "30"},
    });
    composite.inject_audio_and_render(AudioSegment("So, for all intents and purposes, we might as well color this node red too. Red'll win here."));
    consider->highlight = false;
    composite.inject_audio_and_render(AudioSegment("We can repeat this line of thinking for all the other moves which force the game into an ending state."));

    composite.inject_audio(AudioSegment(7), 30);
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
    composite.inject_audio_and_render(AudioSegment("Awesome! But, now, we're out of obvious nodes to fill in."));
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
        {"d", "6"},
    });
    composite.inject_audio_and_render(AudioSegment("What about this one?"));
    
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("There's more than one choice, but both of the children have already been colored with a computed game result."));
    c4.color_edges = true;
    composite.inject_audio_and_render(AudioSegment("If we show the edge colors again, we recall that it's yellow to move in this position."));
    composite.inject_audio_and_render(AudioSegment("Looking at the children, one's yellow's win, and one's a tie."));
    composite.inject_audio_and_render(AudioSegment("Well, Yellow will naturally take the win over a tie."));
    composite.inject_audio_and_render(AudioSegment("Therefore, we can color this node as yellow."));
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
    composite.inject_audio_and_render(AudioSegment(3));

    composite.inject_audio(AudioSegment("In general, a node's status is the best result of all the children, on behalf of the player whose move it is."), 32);
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
    composite.inject_audio_and_render(AudioSegment("The algorithm we just used is called Minimax."));

    g.nodes.at(g.root_node_hash).highlight = true;

    composite.inject_audio_and_render(AudioSegment("We've worked our way all the way back up to the endgame which we considered in the first place, and it turns out that it was a tie!"));

    LatexScene strong_solution_text(VIDEO_WIDTH, VIDEO_HEIGHT/5, "\\text{Strong Solution}", 1);
    composite.add_scene(&strong_solution_text, 0, .8, 1, .2);
    composite.inject_audio_and_render(AudioSegment("This is called a strong solution. In a strong solution, we examine each possible board, and document what the result should be under optimal play."));
    composite.remove_scene(&strong_solution_text);
    composite.inject_audio_and_render(AudioSegment("You can think of the strong solution as a dictionary of positions- you can look up a position, and the dictionary tells you who's favored there."));
    for(auto& node : g.nodes) if(node.second.data->representation == "36426444226456412121132335635611737") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
        {"d", "7"},
    });
    composite.inject_audio_and_render(AudioSegment("If we already have a strong solution, we can use it to play optimally. Let's imagine we're playing as Red."));
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "364264442264564121211323356356117377") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    composite.inject_audio_and_render(AudioSegment("Let's say Yellow plays in the 7th column."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("Now it's our turn, so we look at all the children of the current node, and choose the one with the best outcome."));
    composite.inject_audio_and_render(AudioSegment("In our case here, one child's a tie, and one shows yellow winning."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "3642644422645641212113233563561173777") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("So, we'll choose the tied node, corresponding to a move in the 7th column."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "3642644422645641212113233563561173777") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("Yellow responds in the 5th column."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "36426444226456412121132335635611737775") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("We examine the new children here and play in the 5th column in response, which is the child node with the best outcome."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "364264442264564121211323356356117377755") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("Now yellow plays in column 7."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "3642644422645641212113233563561173777557") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("In this case the children have the same outcome, so we can just pick one at random."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "36426444226456412121132335635611737777755") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("Now, yellow only has one choice, which is to conclude the game as a tie."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    for(auto& node : g.nodes) if(node.second.data->representation == "364264442264564121211323356356117377777555") consider = &(node.second);
    consider->highlight = true;
    board = C4Scene(VIDEO_WIDTH/3, VIDEO_HEIGHT/3, consider->data->representation);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", to_string(consider->x)},
        {"y", to_string(consider->y)},
        {"z", to_string(consider->z)},
    });
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 5;
    composite.inject_audio_and_render(AudioSegment("By doing this, we, as Red, made the best possible move at each turn, and therefore achieved the optimal result for this starting position."));
    for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(consider->neighbors)) const_cast<Edge&>(e).opacity = 1;
    consider->highlight = false;
    composite.remove_scene(&board);


    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "20"},
    });
    composite.inject_audio_and_render(AudioSegment("Now, instead of doing this selection of optimal branches while playing the game, we can do it upfront all at once."));
    for(auto& node : g.nodes) if(node.second.data->representation.size() % 2 == 0){
        node.second.highlight = true;
    }
    composite.inject_audio_and_render(AudioSegment("We consider each node for which it is Red's turn."));
    for(auto& node : g.nodes) if(node.second.data->representation.size() % 2 == 0){
        node.second.highlight = false;
        for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors)) const_cast<Edge&>(e).opacity = 5;
    }
    composite.inject_audio_and_render(AudioSegment("For those nodes, we'll examine their edges, and from there, select an optimal child node."));
    for(auto& node : g.nodes) if(node.second.data->representation.size() % 2 == 0){
        node.second.highlight = false;
        for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors)) const_cast<Edge&>(e).opacity = 1;
        Edge* best_edge;
        int best_status = -1;
        for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors)){
            C4Result res = g.nodes.at(e.to).data->who_is_winning(dummy);
            int status = res==RED?1:(res==TIE?0:-1);
            if(status > best_status){
                best_edge = &(const_cast<Edge&>(e));
                best_status = status;
            }
        }
        best_edge->opacity = 5;
    }
    composite.inject_audio(AudioSegment("We can now delete all the other children."), 6);
    g.mobilize_all_nodes();
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"d", "15"},
        {"y", "0"},
    });
    bool still_an_edge = true;
    while(still_an_edge){
        still_an_edge = false;
        for(auto& node : g.nodes) if(node.second.data->representation.size() % 2 == 0)
            for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors))
                if(e.opacity == 1){
                    g.remove_edge(e.from, e.to);
                    g.delete_orphans();
                    still_an_edge = true;
                    composite.render();
                    break;
                }
    }
    composite.inject_audio_and_render(AudioSegment("It's important to note that this is no longer a strong solution. There are positions which exist but which are no longer present in our graph."));
    composite.inject_audio_and_render(AudioSegment("We've now chopped off all the variations which we wouldn't choose to play as Red."));
    composite.inject_audio_and_render(AudioSegment("In the graph, this means that all the nodes for which it's Red's turn have only one Red edge coming out- namely the optimal one of our selection."));
    for(auto& node : g.nodes)
        for (auto& e : const_cast<std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>&>(node.second.neighbors))
            const_cast<Edge&>(e).opacity = (node.second.data->representation.size() % 2 == 0)?1:5;
    composite.inject_audio_and_render(AudioSegment("But we haven't deleted any of the Yellow edges- you'll still notice that some nodes have multiple yellow children."));
    composite.inject_audio_and_render(AudioSegment("This is because we still need to know how to respond to every possible branch which Yellow can concoct."));
    composite.inject_audio_and_render(AudioSegment("What we've developed here is called a Weak solution (on behalf of Red.) It doesn't contain all positions, but it contains enough for us to guarantee Red to play optimally."));
    composite.inject_audio_and_render(AudioSegment("For that reason it shouldn't be a surprise that all the nodes which were yellow-to-win have been removed."));

    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"y", "-50"},
    });
    composite.inject_audio_and_render(AudioSegment("That would've required a red blunder to get to, and we've removed all branches which contain red blunders."));
}

void minimax_the_opening(){
    std::unordered_map<std::string, std::string> closequat{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "40"},
        {"x", "0"},
        {"y", "-50"},
        {"z", "0"},
        {"surfaces_opacity", "0"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
    };
    auto closequat_moving = closequat;
    closequat_moving["q1"] = "<t> 10 / cos";
    closequat_moving["qj"] = "<t> -10 / sin";
    dag.add_equations(closequat_moving);
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"y", "0"},
    });
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.3;
    g.gravity_strength = 0;
    g.dimensions = 4;
    g.sqrty = true;
    g.lock_root_at_origin = true;
    C4GraphScene c4(&g, "", MANUAL);
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 7; k++){
        g.add_node(new C4Board(to_string(i)));
        g.add_node(new C4Board(to_string(i) + to_string(j)));
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k)));
    }
    c4.inject_audio_and_render(AudioSegment("So the obvious question is, can we apply Minimax to the opening to determine who should win from the starting position?"));
    c4.inject_audio_and_render(AudioSegment("In theory, sure, but even with our modern computers, we can't do this for 4 trillion nodes. It's just too much."));
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"y", "70"},
    });
    c4.inject_audio_and_render(AudioSegment("But by ignoring branches which are suboptimal _while traversing the graph in the first place_, along with a bunch of other tricks, this becomes doable."));
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"y", "-15"},
    });
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"y", "1"},
    });
    g.clear();
    g.add_node(new C4Board(""));
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "5"},
        {"surfaces_opacity", "1"},
    });
    g.dimensions = 2;
    double x = -1.5*4;
    double y = 2;
    for(int i = 1; i <= 7; i++){
        g.add_node_with_position(new C4Board(to_string(i)), (x+=1.5), y, 0);
    }
    g.immobilize_all_nodes();
    CompositeScene openings;
    LatexScene latex_red(VIDEO_WIDTH*.1, VIDEO_HEIGHT*.1, "\\text{\\textcolor{red}{Red}}", 1);
    LatexScene latex_tie(VIDEO_WIDTH*.1, VIDEO_HEIGHT*.1, "\\text{\\textcolor{blue}{Tie}}", 1);
    LatexScene latex_yel(VIDEO_WIDTH*.1, VIDEO_HEIGHT*.1, "\\text{\\textcolor{yellow}{Yellow}}", 1);
    openings.add_scene(&c4, 0, 0, 1, 1);
    openings.inject_audio_and_render(AudioSegment("Using these strategies, Connect 4 was solved."));
    openings.add_scene(&latex_red, .45, .8, .1, .1);
    double d = .1275;
    openings.inject_audio_and_render(AudioSegment("We know that under optimal play, Red Wins, and only does so by starting with a disk in the center column."));
    openings.add_scene(&latex_tie, .45-1*d, .8, .1, .1);
    openings.add_scene(&latex_tie, .45+1*d, .8, .1, .1);
    openings.inject_audio_and_render(AudioSegment("One space off to either side and it's a tie,"));
    openings.add_scene(&latex_yel, .45-3*d, .8, .1, .1);
    openings.add_scene(&latex_yel, .45-2*d, .8, .1, .1);
    openings.add_scene(&latex_yel, .45+2*d, .8, .1, .1);
    openings.add_scene(&latex_yel, .45+3*d, .8, .1, .1);
    openings.inject_audio_and_render(AudioSegment("and anywhere else, it's Yellow to win."));

    LatexScene latex_qn(VIDEO_WIDTH, VIDEO_HEIGHT, "\\text{...?}", 1);
    openings.add_scene(&latex_qn, 0, 0, 1, 1);
    openings.inject_audio_and_render(AudioSegment("But why?"));
    openings.inject_audio_and_render(AudioSegment("Knowing the game-theoretically-optimal move at any position's great, but it provides absolutely no insight into why this is the case."));
}

void prisoner() {
    FOR_REAL = false;
    ThreeDimensionScene tds;

    LatexScene flashcard1_front("(157, 286)", .75);
    LatexScene flashcard1_back("\\text{60a5be}", .75);
    LatexScene flashcard2_front("(483, 27)", .75);
    LatexScene flashcard2_back("\\text{81b0b9}", .75);
    LatexScene flashcard3_front("(202, 105)", .75);
    LatexScene flashcard3_back("\\text{ef9229}", .75);
    Pixels roundrect(flashcard1_front.w, flashcard1_front.h);
    roundrect.rounded_rect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, 100, 0xff222222);
    flashcard1_front.expose_pixels()->underlay(roundrect, 0, 0);
    flashcard1_back.expose_pixels()->underlay(roundrect, 0, 0);
    flashcard2_front.expose_pixels()->underlay(roundrect, 0, 0);
    flashcard2_back.expose_pixels()->underlay(roundrect, 0, 0);
    flashcard3_front.expose_pixels()->underlay(roundrect, 0, 0);
    flashcard3_back.expose_pixels()->underlay(roundrect, 0, 0);

    tds.add_surface(Surface(glm::vec3(0,0  , 0.01),glm::vec3( 1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&flashcard1_front));
    tds.add_surface(Surface(glm::vec3(0,0  ,-0.01),glm::vec3(-1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&flashcard1_back));
    tds.add_surface(Surface(glm::vec3(0,100, 0.01),glm::vec3( 1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&flashcard2_front));
    tds.add_surface(Surface(glm::vec3(0,100,-0.01),glm::vec3(-1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&flashcard2_back));
    tds.add_surface(Surface(glm::vec3(0,200, 0.01),glm::vec3( 1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&flashcard3_front));
    tds.add_surface(Surface(glm::vec3(0,200,-0.01),glm::vec3(-1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&flashcard3_back));

    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "4"},
    });
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"ntf", "<subscene_transition_fraction> .5 -"},
        {"x", "3 <ntf> * 4 ^ -1 *"},
        {"sigmoid", "3.1415 4 / 1 2.71828 <ntf> 40 * ^ + /"},
        {"q1", "<sigmoid> cos"},
        {"qi", "0"},
        {"qj", "<sigmoid> sin"},
        {"qk", "0"}
    });
    tds.inject_audio(AudioSegment("Lemme make an analogy. Imagine a prisoner is forced to memorize a deck of flash cards. On the front of each card is a 2-dimensional coordinate, and the back of the flash card is a hexadecimal number."), 3);
    tds.render();
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"y", "100"}
    });
    tds.render();
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"y", "200"}
    });
    tds.render();



    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "2 <transition_fraction> 8 * ^"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"}
    });
    ThreeDimensionScene cardgrid;
    for(int x = -22; x < 22; x++){
        for(int y = -59; y < 59; y++){
            Scene* sc = ((x+y)%3 == 0) ? &flashcard1_back : (((x+y)%3 == 1) ? &flashcard2_back : &flashcard3_back);
            cardgrid.add_surface(Surface(glm::vec3((x+.5)*2.*VIDEO_WIDTH/VIDEO_HEIGHT,(y+.5)*2,0),glm::vec3(1, 0, 0),glm::vec3(0, static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),sc));
        }
    }
    PngScene mona_lisa("Mona_Lisa_Hands");
    Surface s(glm::vec3(0,0,-1),glm::vec3(mona_lisa.w*.1, 0, 0),glm::vec3(0, mona_lisa.h*.1, 0),&mona_lisa);
    cardgrid.add_surface(s);
    cardgrid.inject_audio(AudioSegment("The prison guard forgot to mention that the color codes on the back of the cards document the precise color of the Mona Lisa painting at those two coordinates, and, sadly, our prisoner was too caught up memorizing numbers to realize the structure underlying the data."), 100);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < cardgrid.surfaces.size(); j++){
            cardgrid.surfaces[j].opacity = 1-i/100.;
        }
        cardgrid.surfaces[cardgrid.surfaces.size()-1].opacity = cube(i/100.);
        cardgrid.render();
    }
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"d", "256"},
    });
    for(int j = 0; j < cardgrid.surfaces.size() - 1; j++){
        cardgrid.surfaces[j].opacity = 0;
    }
    cardgrid.inject_audio_and_render(AudioSegment("You may call that prisoner the worlds leading expert on the Mona Lisa..."));
    cardgrid.surfaces.clear();
    cardgrid.add_surface(s);
    ExposedPixelsScene black_screen;
    black_screen.expose_pixels()->fill(OPAQUE_BLACK);

    int count = 0;
    for(int y = 12; y >= -12; y--){
        for(int x = 5; x >= -5; x--){
            cardgrid.add_surface(Surface(glm::vec3((x+.5)*10.*VIDEO_WIDTH/VIDEO_HEIGHT,(y+.5)*10,0),glm::vec3(10, 0, 0),glm::vec3(0, 10*static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH, 0),&black_screen));
            count++;
        }
    }
    cardgrid.inject_audio(AudioSegment("At least in the sense that, if somebody held a contest to recreate the painting from memory, he'd win it, pixel-by-pixel. As long as the prison guard remembers to tell him that's what the data represents."), count);
    for(int x = 0; x < count; x++){
        cardgrid.surfaces.pop_back();
        cardgrid.render();
    }

    PngScene mona_lisa_no_hands("Mona_Lisa");
    Surface no_hands(glm::vec3(0,0,-1.1),glm::vec3(mona_lisa_no_hands.w*.1, 0, 0),glm::vec3(0, mona_lisa_no_hands.h*.1, 0),&mona_lisa_no_hands);
    cardgrid.add_surface(no_hands);
    cardgrid.inject_audio(AudioSegment("But until that happens, if you were to go up to him and ask 'Is Mona Lisa's left hand on top of her right, or is it the other way around?', he would have no idea."), 100);
    for(int x = 0; x < 100; x++){
        glm::vec3 dir(1,0,0);
        cardgrid.surfaces[cardgrid.surfaces.size()-1].center += dir;
        cardgrid.surfaces[0].center -= dir;
        cardgrid.render();
    }

    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.4;
    g.dimensions = 3;
    C4GraphScene c4(&g, "", MANUAL);
    c4.physics_multiplier = 1;
    if(FOR_REAL)for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 7; k++)for(int l = 1; l <= 7; l++){
        g.add_node(new C4Board(to_string(i)));
        g.add_node(new C4Board(to_string(i) + to_string(j)));
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k)));
        g.add_node(new C4Board(to_string(i) + to_string(j) + to_string(k) + to_string(l)));
    }
    g.iterate_physics(100);
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"y", "200"},
    });

    CompositeScene composite;
    composite.add_scene(&c4      , 0, 0, 1, 1);
    composite.add_scene(&cardgrid, 0, 0, 1, 1);

    glm::vec3 dir(0,200,0);
    cardgrid.surfaces[0].center += dir;
    cardgrid.surfaces[1].center += dir;
    c4.skip_surfaces = true;
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"y", "190"},
        {"q1", "1"},
        {"qi", "-.1"},
        {"qj", "0"},
        {"qk", "0"},
    });
    composite.inject_audio_and_render(AudioSegment("This analogy sort of illustrates the position we're in with our strong solution for connect 4."));
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"y", "0"},
        {"d", "70"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });
    composite.inject_audio_and_render(AudioSegment("Just because we know these disparate facts about who wins in what scenario, we can't say we truly understand this system any better than before."));
    dag.add_transitions(std::unordered_map<std::string, std::string>{
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"},
    });
    c4.inject_audio_and_render(AudioSegment("What's more, this information is effectively incommunicable. Any solution tree is so big that you simply can't be bestowed that information."));
    c4.inject_audio_and_render(AudioSegment("You can't realistically visualize it. You can't realistically memorize it."));
}

void render_tree_comparison(){
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"},
        {"d", "10"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "0"},
        {"lines_opacity", "1"},
        {"points_opacity", "1"},
    });


    std::vector<string> starting_positions = {"4444443265555232253333727111177771125","44444432655552322533337271111777711","444444326555523225333372711117777","4444443265555232253333727111177",};
    std::vector<string> scripts = {
        "This position is pretty simple- the game is gonna end in 2 moves, and there are no choices to be made. Naturally, our graph is also extremely simple. Let's delete some discs from the board and see how the graphs grow.",
        "Ok, that's already getting a little bit bigger. I'll keep plucking off pieces and you can watch what happens.",
        "All of the graphs still grow exponentially.",
        "This should give some intuition that even if you claimed to have memorized a _weak_ solution tree of the whole game, merely verifying your memorization would likely take a lifetime.",
    };

    if(FOR_REAL)for(int i = 0; i < starting_positions.size(); i++){
        string starting_position = starting_positions[i];
        C4Scene board(VIDEO_WIDTH/2, VIDEO_HEIGHT/2, starting_position);
        LatexScene board_header(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.1, "\\text{Position} \\\\\\\\ \\text{" + to_string(starting_position.size()) + " discs placed}", 1);

        Graph<C4Board> strong;
        strong.decay = 0.1;
        strong.repel_force = 1;
        strong.dimensions = 3;
        C4GraphScene strong_scene(VIDEO_WIDTH/2, VIDEO_HEIGHT/2, &strong, starting_position, FULL);
        LatexScene strong_header(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.1, "\\text{Strong Solution / Full Graph} \\\\\\\\ \\text{" + to_string(strong.size()) + " nodes}", 1);

        Graph<C4Board> weak;
        weak.decay = 0.3;
        weak.repel_force = 1;
        weak.dimensions = 3;
        C4GraphScene weak_scene(VIDEO_WIDTH/2, VIDEO_HEIGHT/2, &weak, starting_position, SIMPLE_WEAK);
        LatexScene weak_header(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.1, "\\text{The Smallest Weak Solution} \\\\\\\\ \\text{" + to_string(weak.size()) + " nodes}", 1);

        Graph<C4Board> union_weak;
        union_weak.decay = 0.1;
        union_weak.repel_force = 1;
        union_weak.dimensions = 3;
        C4GraphScene union_weak_scene(VIDEO_WIDTH/2, VIDEO_HEIGHT/2, &union_weak, starting_position, UNION_WEAK);
        LatexScene union_weak_header(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.1, "\\text{Union of All Weak Solutions} \\\\\\\\ \\text{" + to_string(union_weak.size()) + " nodes}", 1);

        CompositeScene composite;
        composite.add_scene(&board            , 0  , 0  , 0.5, 0.5);
        composite.add_scene(&strong_scene     , 0  , 0.5, 0.5, 0.5);
        composite.add_scene(&weak_scene       , 0.5, 0  , 0.5, 0.5);
        composite.add_scene(&union_weak_scene , 0.5, 0.5, 0.5, 0.5);
        composite.add_scene(&board_header     , 0.1, 0.4, 0.3, 0.1);
        composite.add_scene(&strong_header    , 0.1, 0.9, 0.3, 0.1);
        composite.add_scene(&weak_header      , 0.6, 0.4, 0.3, 0.1);
        composite.add_scene(&union_weak_header, 0.6, 0.9, 0.3, 0.1);

        composite.inject_audio_and_render(AudioSegment(scripts[i]));
        if(i == starting_positions.size()-1){
            composite.inject_audio_and_render(AudioSegment("So, is there any way to rigorously unite the wisdom and intuition accrued by human experts with the output of this enormous algorithm?"));
            composite.inject_audio_and_render(AudioSegment("Well, a strong solution tells us everything that could be known, but weak solutions have a lot more room for creative expression."));

            ExposedPixelsScene black_screen(VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
            black_screen.expose_pixels()->fill(0xcc000000);
            composite.add_scene(&black_screen, 0  , 0  , 0.5, 0.5);
            composite.add_scene(&black_screen, 0  , 0.5, 0.5, 0.5);
            composite.add_scene(&black_screen, 0.5, 0.5, 0.5, 0.5);
            composite.inject_audio_and_render(AudioSegment("Take a close look at this particular weak solution- notice how it has a very regular structure?"));
        }
    }
    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"points_opacity", "0"},
    });

    if(FOR_REAL){
        dag.add_equations(std::unordered_map<std::string, std::string>{
            {"d", "80"},
        });
        string starting_position = "444444326555523225333372711";
        C4Scene board(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.3, starting_position);
        Graph<C4Board> weak;
        weak.decay = 0.3;
        weak.repel_force = 2;
        weak.dimensions = 3;
        C4GraphScene weak_scene(&weak, starting_position, SIMPLE_WEAK);
        weak_scene.physics_multiplier = 5;
        CompositeScene composite;
        composite.add_scene(&weak_scene, 0, 0, 1  , 1  );
        composite.add_scene(&board     , 0, 0, 0.3, 0.3);

        composite.inject_audio_and_render(AudioSegment("We can remove even more pieces and get a better view..."));
        composite.inject_audio_and_render(AudioSegment("Y'know how when your computer is storing files, structured information can easily be compressed and the filesize can be decreased?"));
    }

    if(FOR_REAL){
        dag.add_equations(std::unordered_map<std::string, std::string>{
            {"d", "120"},
        });
        string starting_position = "4444443265555232253333727";
        C4Scene board(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.3, starting_position);
        Graph<C4Board> weak;
        weak.decay = 0.3;
        weak.repel_force = 2;
        weak.dimensions = 3;
        C4GraphScene weak_scene(&weak, starting_position, SIMPLE_WEAK);
        weak_scene.physics_multiplier = 5;
        CompositeScene composite;
        composite.add_scene(&weak_scene, 0, 0, 1  , 1  );
        composite.add_scene(&board     , 0, 0, 0.3, 0.3);

        composite.inject_audio_and_render(AudioSegment("Can we immensely reduce the amount of information required to 'know' a weak solution, such that it doesn't require rote memorization?"));
        composite.inject_audio_and_render(AudioSegment("In other words, are there any particularly clever ways of representing such weak solutions? We can represent them graph-theoretically, but perhaps they can be expressed by other means?"));
    }

    {
        FOR_REAL = true;
        dag.add_equations(std::unordered_map<std::string, std::string>{
            {"q1", "1"},
            {"qi", "<t> -4 / cos"},
            {"qj", "<t> -4 / sin"},
            {"qk", "0"},
        });
        ss_list = std::array<std::string, C4_HEIGHT>{
            "  22@| ",
            "  112| ",
            " 1221| ",
            " 2112| ",
            " 2121|@",
            " 211211"
        };
        ss_simple_weak = SteadyState(ss_list);
        dag.add_equations(std::unordered_map<std::string, std::string>{
            {"d", "180"},
        });
        string starting_position = "44444432655552322533337";
        C4Scene board(VIDEO_WIDTH*.3, VIDEO_HEIGHT*.3, starting_position);
        Graph<C4Board> weak;
        weak.decay = 0.3;
        weak.repel_force = 2;
        weak.dimensions = 3;
        weak.gravity_strength = 0.1;
        C4GraphScene weak_scene(&weak, starting_position, SIMPLE_WEAK);
        weak_scene.physics_multiplier = 5;
        CompositeScene composite;
        composite.add_scene(&weak_scene, 0, 0, 1  , 1  );
        composite.add_scene(&board     , 0, 0, 0.3, 0.3);

        composite.inject_audio_and_render(AudioSegment("Can we immensely reduce the amount of information required to 'know' a weak solution, such that it doesn't require rote memorization?"));
        dag.add_transitions(std::unordered_map<std::string, std::string>{
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "1"},
        });
        composite.inject_audio_and_render(AudioSegment("In other words, are there any particularly clever ways of representing such weak solutions? We can represent them graph-theoretically, but perhaps they can be expressed by other means?"));
        weak_scene.skip_surfaces = true;

        dag.add_transitions(std::unordered_map<std::string, std::string>{
            {"d", "270"},
            {"surfaces_opacity", "1"},
            {"lines_opacity", "0.2"},
            {"points_opacity", "0"},
            {"q1", "1"},
            {"qi", "<t> sin 400 /"},
            {"qj", "<t> sin 400 /"},
            {"qk", "0"},
        });

        CompositeScene shill_compositely;
        ThreeDimensionScene shill;
        PngScene claimeven("Claimeven_Thumb");
        LatexScene words1("\\text{If you haven't already seen my Claimeven video...}", 1);
        LatexScene words2("\\text{Check it out now!}", 1);
        double t = 1.2; Surface c (glm::vec3(20 ,0 ,-240),glm::vec3(claimeven.w*.005*sin(t), claimeven.w*.005*cos(t), 0),glm::vec3(claimeven.h*.005*-cos(t), claimeven.h*.005*sin(t), 0),&claimeven);
               t = 1.8; Surface w1(glm::vec3(-12,-5,-235),glm::vec3(words1.w   *.04 *sin(t), words1.w   *.04 *cos(t), 0),glm::vec3(words1.h   *.04 *-cos(t), words1.h   *.04 *sin(t), 0),&words1   );
               t = 1.4; Surface w2(glm::vec3(-10,5 ,-245),glm::vec3(words2.w   *.02 *sin(t), words2.w   *.02 *cos(t), 0),glm::vec3(words2.h   *.02 *-cos(t), words2.h   *.02 *sin(t), 0),&words2   );
        shill.add_surface(c );
        shill.add_surface(w1);
        shill.add_surface(w2);
        shill_compositely.add_scene(&composite, 0, 0, 1, 1);
        shill_compositely.add_scene(&shill    , 0, 0, 1, 1);
        shill_compositely.inject_audio_and_render(AudioSegment("I'll give you a hint- Claimeven is the magic bullet here!"));
        shill_compositely.inject_audio_and_render(AudioSegment("We can use a positional language built on top of the idea of claimeven to concisely express solutions to positions which have a high degree of self-symmetry,"));
        dag.add_transitions(std::unordered_map<std::string, std::string>{
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "0"},
            {"d", "180"},
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"surfaces_opacity", "1"},
            {"lines_opacity", "1"},
            {"points_opacity", "0"},
        });
        shill_compositely.inject_audio_and_render(AudioSegment("and in doing so, we can get these graphs down from the trillions of nodes to the order of the tens of thousands."));
    }

/*
    .inject_audio_and_render(AudioSegment("Or at least, I think so... I still haven't been able to perform this reduction for the entire graph. Still workin on that technical challenge."));
    .inject_audio_and_render(AudioSegment("But, what I can tell you for sure is that there's some real human openings which, using reduction by claimeven-like strategies, have weak solutions expressable with less than a thousand nodes. Not even a megabyte."));
    .inject_audio_and_render(AudioSegment("And it's TOTALLY visualizable, and, if you wanna go there, memorizable too."));
    .inject_audio_and_render(AudioSegment("I'll spoil two of them here, but you'll have to stay tuned to see exactly how we accomplish this, as well as what we can learn about systems other than connect 4 from these compressed solutions."));
    .inject_audio_and_render(AudioSegment("This has been 2swap."));
*/
}

void render_video() {
    FOR_REAL = false;
    PRINT_TO_TERMINAL = false;
    if(FOR_REAL){
        beginning();
        endgame_examination();
        minimax_the_opening();
        prisoner();
    }
    render_tree_comparison();
}