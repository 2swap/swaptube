using namespace std;
void render_video() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.4;
    g.gravity_strength = 0;
    g.dimensions = 2;
    g.sqrty = true;
    g.lock_root_at_origin = true;
    C4GraphScene c4(&g, "444", MANUAL);
    c4.physics_multiplier = 1;
    g.sanitize_for_closure();

    PRINT_TO_TERMINAL = false;
    std::unordered_map<std::string, std::string> closequat{
        {"q1", "t 4 / cos"},
        {"qi", "0"},
        {"qj", "t -4 / sin"},
        {"qk", "0"},
        {"d", "2"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "1"},
    };
    dag.add_equations(closequat);
    c4.inject_audio_and_render(AudioSegment("So far we have been interested in strategies of connect 4,"));
    dag.add_equation("d", "20");
    c4.inject_audio_and_render(AudioSegment("but now it's time to apply those strategies to actually get a structural understanding of the game."));

    /*
    if(FOR_REAL){
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board("444" + to_string(i)));
        g.sanitize_for_closure();
        v.inject_audio_and_render(AudioSegment(.1));
    }
    g.dimensions = 3;
    for(int i = 1; i <= 7; i++){
        for(int j = 1; j <= 7; j++){
            if(j==4) continue;
            g.add_node(new C4Board("444" + to_string(i) + to_string(j)));
            g.sanitize_for_closure();
            v.inject_audio_and_render(AudioSegment(.1));
        }
    }
    }
    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "0"}
    });
    v.inject_audio_and_render(AudioSegment("First of all, you're gonna have to get comfortable imagining the game as a tree."));

    v.stage_transition(closequat);


    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"y", "13"}
    });
    v.inject_audio_and_render(AudioSegment(1));
    std::vector<double> nodes_to_remove;
    for (const auto& node_it : g.nodes) {
        nodes_to_remove.push_back(node_it.first);
    }
    for (const double key : nodes_to_remove) {
        g.remove_node(key);
    }


    v.set_variables(std::unordered_map<std::string, std::string>{
        {"y", "1.5"}, {"d", "10"},
        {"q1", "0"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });
    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"}
    });

    g.add_node(new C4Board("436"));
    
    
    
    
    v.inject_audio_and_render(AudioSegment("For any starting position, we can draw a node."));
    g.gravity_strength = 1;
    c4.physics_multiplier = 0;
    g.dimensions = 2;
    c4.inject_audio_and_render(AudioSegment("there is some amount of moves that can be made from that position, which we will represent as nodes connected to the root."));

    double x = -1.5*4;
    double y = 2;
    g.add_node_with_position(new C4Board("4361"), (x+=1.5), y, 0);
    g.sanitize_for_closure();
    v.inject_audio_and_render(AudioSegment("Yellow can play in the left column,"));

    g.add_node_with_position(new C4Board("4362"), (x+=1.5), y, 0);
    g.sanitize_for_closure();
    v.inject_audio_and_render(AudioSegment("the next column over,"));

    v.inject_audio(AudioSegment("and so on successively."), 5);
    for(int i = 3; i <= 7; i++){
        g.add_node_with_position(new C4Board("436" + to_string(i)), (x+=1.5), y, 0);
        g.sanitize_for_closure();
        v.render();
    }
    g.immobilize_all_nodes();

    c4.physics_multiplier = 1;
    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"d", "10 t 2 / +"}
    });
    v.inject_audio_and_render(AudioSegment("We connect those new nodes to the root with yellow lines, since yellow was the one who played a move here."));
    v.inject_audio(AudioSegment("And from those positions, there are even more options that can be made, by Red this time, continuing the graph."), 49);
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++){
        g.add_node_with_position(new C4Board("436" + to_string(i) + to_string(j)), (i-4)*1.5, 4, 0);
        g.sanitize_for_closure();
        v.render();
    }








    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"d", "30"}
    });
    v.inject_audio(AudioSegment("In other words, every path through this graph represents a particular continuation from the existing board."), 4);
    for(Surface& s : c4.surfaces) s.opacity = 0.2;
    for(Surface& s : c4.surfaces) s.opacity = 0.2;
    for(Line& l : c4.lines) l.opacity = 0.2;
    v.render();
    for(Surface& s : c4.surfaces) s.opacity = string("43667").find(s.name) != string::npos? 1 : 0.2;
    for(Line& l : c4.lines) l.opacity = l.name == "4366 - 43667" || l.name != "436 - 4366" ? 1 : 0.2;
    v.render();
    for(Surface& s : c4.surfaces) s.opacity = string("43676").find(s.name) != string::npos? 1 : 0.2;
    for(Line& l : c4.lines) l.opacity = l.name == "4367 - 43676" || l.name != "436 - 4367" ? 1 : 0.2;
    v.render();
    for(Surface& s : c4.surfaces) s.opacity = string("43655").find(s.name) != string::npos? 1 : 0.2;
    for(Line& l : c4.lines) l.opacity = l.name == "4365 - 43655" || l.name != "436 - 4365" ? 1 : 0.2;
    v.render();
    for(Surface& s : c4.surfaces) s.opacity = 1;
    for(Line& l : c4.lines) l.opacity = 1;
    v.inject_audio_and_render(AudioSegment("Any set of moves that you could make is in this graph."));

    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"x", "10"}
    });
    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "0"}
    });
    v.inject_audio_and_render(AudioSegment("As a result, it gets intractably large, really fast."));
    g.dimensions = 3;
    g.gravity_strength = 0;
    g.mobilize_all_nodes();
    v.stage_transition(closequat);
    v.inject_audio_and_render(AudioSegment("At any position in the opening, there are 7 possible moves to make, This means,"));
    for(int i = 1; i <= 7; i++)for(int j = 1; j <= 7; j++)for(int k = 1; k <= 2; k++){
        g.add_node(new C4Board("436" + to_string(i) + to_string(j) + to_string(k)));
        g.sanitize_for_closure();
        v.render();
    }
    v.inject_audio_and_render(AudioSegment("at a depth of 0, the amount of nodes is 1 (the empty board),"));
    v.inject_audio_and_render(AudioSegment("at a depth of 1, the amount of nodes is 7 (the 7 openings),"));
    v.inject_audio_and_render(AudioSegment("at a depth of 2, the amount of nodes is 49,"));
    /*
    v.inject_audio_and_render(AudioSegment("and the progression continues exponentially."));
    v.inject_audio_and_render(AudioSegment("In fact, John Tromp determined that there are a total of 4,531,985,219,092 unique connect 4 positions."));
    v.inject_audio_and_render(AudioSegment("We can do some tricks like de-duplicating based on horizontal symmetry,"));
    v.inject_audio_and_render(AudioSegment("but the full game graph is still incomprehensibly huge, tangled, and complex."));
    v.inject_audio_and_render(AudioSegment("Let's start with an endgame position to make life a little bit easier on ourselves."));
    v.inject_audio_and_render(AudioSegment("This is a particular endgame, and here is what it looks like when we expand out its full tree of positions."));
    v.inject_audio_and_render(AudioSegment("Now, let's color in all the terminal states of this board."));
    v.inject_audio_and_render(AudioSegment("In other words, we are identifying the games which have either been won for red,"));
    v.inject_audio_and_render(AudioSegment("won for yellow,"));
    v.inject_audio_and_render(AudioSegment("or have completely filled without a winner as a tie."));
    v.inject_audio_and_render(AudioSegment("Now, this is more easy to tell what's going on."));
    v.inject_audio_and_render(AudioSegment("In the case of this endgame, there are some paths that lead to a red victory,"));
    v.inject_audio_and_render(AudioSegment("some paths that lead to a Yellow victory,"));
    v.inject_audio_and_render(AudioSegment("and some paths that lead to the game becoming a tie."));
    v.inject_audio_and_render(AudioSegment("So, who is favored in the root position?"));
    v.inject_audio_and_render(AudioSegment("In other words, if both players play perfectly, who wins?"));
    v.inject_audio_and_render(AudioSegment("Well, in this board Red has already won, but let's take a look at the board before it, in which Red is just about to win."));
    v.inject_audio_and_render(AudioSegment("In this board, Red has the option to claim victory right away,"));
    v.inject_audio_and_render(AudioSegment("but Red also has the option to, for example, not notice it whatsoever and play this move, thereby permitting a Yellow victory."));
    v.inject_audio_and_render(AudioSegment("we want to know who would win under perfect play, so we will assume Red doesn't make this mistake, and succeeds in taking the correct move."));
    v.inject_audio_and_render(AudioSegment("So, we can color this node as red too, since Red should win under perfect play."));
    v.inject_audio_and_render(AudioSegment("In this case, Red has the option to either block Yellow on the top row, permitting the board to fill up as a tie."));
    v.inject_audio_and_render(AudioSegment("Alternately, Red can not do so, letting Yellow win."));
    v.inject_audio_and_render(AudioSegment("Obviously, Red will block."));
    v.inject_audio_and_render(AudioSegment("In general, if it's Red move, red will pick the most favorable continuation. If there is a Red child node, then Red will take that at first priority, meaning the current board is red-to-win."));
    v.inject_audio_and_render(AudioSegment("If there isn't one, but there is a tying child node, then Red would prefer that to a Yellow-win child node, making the current board a tie too."));
    v.inject_audio_and_render(AudioSegment("If there's nothing but Yellow continuations, then tough luck... the current board is in the bag for Yellow, since no matter what red does, Yellow is in the lead."));
    v.inject_audio_and_render(AudioSegment("So, for all of the boards in this graph for which it is red's turn, then we can work back given this algorithm and figure out who is winning."));
    v.inject_audio_and_render(AudioSegment("We can do the exact same decision procedure for Yellow, but in reverse. Yellow prioritizes yellow continuations if they exist, then tied continuations, and then red continuations."));
    v.inject_audio_and_render(AudioSegment("Given this simple decision procedure, we can work back all the way to our starting node, coloring in all of the nodes along the way."));
    v.inject_audio_and_render(AudioSegment("This way, we know that, if both players play perfectly, Red will win."));
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
    v.inject_audio_and_render(AudioSegment("This is great and all, but it provides absolutely no information into why, or how, this is the case. "));
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
