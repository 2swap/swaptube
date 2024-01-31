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
    VariableScene v(&c4);

    std::unordered_map<std::string, std::string> closequat{
        {"q1", "t 4 / cos"},
        {"qi", "0"},
        {"qj", "t -4 / sin"},
        {"qk", "0"},
        {"d", "2"}
    };
    FOR_REAL = false;
    v.set_variables(closequat);
    v.inject_audio_and_render(AudioSegment("So far we have been interested in strategies of connect 4,"));
    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"q1", "t 4 / cos"},
        {"qi", "0"},
        {"qj", "t -4 / sin"},
        {"qk", "0"},
        {"d", "20"}
    });
    v.inject_audio_and_render(AudioSegment("but now it's time to apply those strategies to actually get a structural understanding of the game."));

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
    FOR_REAL = true;
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
        {"y", "0"}, {"d", "10"}
    });


    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"}
    });

    g.add_node(new C4Board("444"));
    v.inject_audio_and_render(AudioSegment("For any starting position, we can draw a node."));










    
    FOR_REAL = false;
    c4.inject_audio_and_render(AudioSegment("there is some amount of moves that can be made from that position, which we will represent as nodes connected to the root."));
    c4.inject_audio_and_render(AudioSegment("We'll connect those new nodes to the root with red lines, since red was the one who made a move here."));
    c4.inject_audio_and_render(AudioSegment("And from those positions, there are even more options that can be made, by Yellow this time, continuing the graph."));
    c4.inject_audio_and_render(AudioSegment("In other words, every path through this graph represents a particular continuation from the existing board."));
    c4.inject_audio_and_render(AudioSegment("Some of them are really stupid continuations."));
    c4.inject_audio_and_render(AudioSegment("Some of them are the most commonly played lines by human players."));
    c4.inject_audio_and_render(AudioSegment("Any set of moves that you could make is in this graph."));
    c4.inject_audio_and_render(AudioSegment("As a result, it gets intractably large, really fast."));
    c4.inject_audio_and_render(AudioSegment("At any position in the opening, there are 7 possible moves to make, This means,"));
    c4.inject_audio_and_render(AudioSegment("at a depth of 0, the amount of nodes is 1 (the empty board),"));
    c4.inject_audio_and_render(AudioSegment("at a depth of 1, the amount of nodes is 7 (the 7 openings),"));
    c4.inject_audio_and_render(AudioSegment("at a depth of 2, the amount of nodes is 49,"));
    c4.inject_audio_and_render(AudioSegment("Here's all the nodes of depth 3, 4, 5, ..."));
    c4.inject_audio_and_render(AudioSegment("and the progression continues exponentially."));
    c4.inject_audio_and_render(AudioSegment("In fact, John Tromp determined that there are a total of 4,531,985,219,092 unique connect 4 positions."));
    c4.inject_audio_and_render(AudioSegment("We can do some tricks like de-duplicating based on horizontal symmetry,"));
    c4.inject_audio_and_render(AudioSegment("but the full game graph is still incomprehensibly huge, tangled, and complex."));
    c4.inject_audio_and_render(AudioSegment("Let's start with an endgame position to make life a little bit easier on ourselves."));
    c4.inject_audio_and_render(AudioSegment("This is a particular endgame, and here is what it looks like when we expand out its full tree of positions."));
    c4.inject_audio_and_render(AudioSegment("Now, let's color in all the terminal states of this board."));
    c4.inject_audio_and_render(AudioSegment("In other words, we are identifying the games which have either been won for red,"));
    c4.inject_audio_and_render(AudioSegment("won for yellow,"));
    c4.inject_audio_and_render(AudioSegment("or have completely filled without a winner as a tie."));
    c4.inject_audio_and_render(AudioSegment("Now, this is more easy to tell what's going on."));
    c4.inject_audio_and_render(AudioSegment("In the case of this endgame, there are some paths that lead to a red victory,"));
    c4.inject_audio_and_render(AudioSegment("some paths that lead to a Yellow victory,"));
    c4.inject_audio_and_render(AudioSegment("and some paths that lead to the game becoming a tie."));
    c4.inject_audio_and_render(AudioSegment("So, who is favored in the root position?"));
    c4.inject_audio_and_render(AudioSegment("In other words, if both players play perfectly, who wins?"));
    c4.inject_audio_and_render(AudioSegment("Well, in this board Red has already won, but let's take a look at the board before it, in which Red is just about to win."));
    c4.inject_audio_and_render(AudioSegment("In this board, Red has the option to claim victory right away,"));
    c4.inject_audio_and_render(AudioSegment("but Red also has the option to, for example, not notice it whatsoever and play this move, thereby permitting a Yellow victory."));
    c4.inject_audio_and_render(AudioSegment("we want to know who would win under perfect play, so we will assume Red doesn't make this mistake, and succeeds in taking the correct move."));
    c4.inject_audio_and_render(AudioSegment("So, we can color this node as red too, since Red should win under perfect play."));
    c4.inject_audio_and_render(AudioSegment("In this case, Red has the option to either block Yellow on the top row, permitting the board to fill up as a tie."));
    c4.inject_audio_and_render(AudioSegment("Alternately, Red can not do so, letting Yellow win."));
    c4.inject_audio_and_render(AudioSegment("Obviously, Red will block."));
    c4.inject_audio_and_render(AudioSegment("In general, if it's Red move, red will pick the most favorable continuation. If there is a Red child node, then Red will take that at first priority, meaning the current board is red-to-win."));
    c4.inject_audio_and_render(AudioSegment("If there isn't one, but there is a tying child node, then Red would prefer that to a Yellow-win child node, making the current board a tie too."));
    c4.inject_audio_and_render(AudioSegment("If there's nothing but Yellow continuations, then tough luck... the current board is in the bag for Yellow, since no matter what red does, Yellow is in the lead."));
    c4.inject_audio_and_render(AudioSegment("So, for all of the boards in this graph for which it is red's turn, then we can work back given this algorithm and figure out who is winning."));
    c4.inject_audio_and_render(AudioSegment("We can do the exact same decision procedure for Yellow, but in reverse. Yellow prioritizes yellow continuations if they exist, then tied continuations, and then red continuations."));
    c4.inject_audio_and_render(AudioSegment("Given this simple decision procedure, we can work back all the way to our starting node, coloring in all of the nodes along the way."));
    c4.inject_audio_and_render(AudioSegment("This way, we know that, if both players play perfectly, Red will win."));
    c4.inject_audio_and_render(AudioSegment("To know the strategy that Red must employ to win, we simply move along the graph by only selecting nodes which are colored red, to guarantee we arrive at a red win."));
    c4.inject_audio_and_render(AudioSegment("We can change our coloring scheme to show who is winning on a certain path."));
    c4.inject_audio_and_render(AudioSegment("This algorithm for determining who is winning is called Minimax."));
    c4.inject_audio_and_render(AudioSegment("So the immediate question then becomes, can we apply it to the opening to determine who should win for the empty board?"));
    c4.inject_audio_and_render(AudioSegment("In theory, sure, but even with our modern computers, we can't do this for 4 trillion nodes. It's just too much."));
    c4.inject_audio_and_render(AudioSegment("But there are a number of clever tricks that you can do."));
    c4.inject_audio_and_render(AudioSegment("If we already determined that, on a Red move, there is a single Red child, then there is no need to search the remaining variations. We can simply discard them."));
    c4.inject_audio_and_render(AudioSegment("This turns out to be really big, because at all resolutions of the exponential growth of the game tree, you are shaving off entire enormous branches of the game tree which don't need to be evaluated."));
    c4.inject_audio_and_render(AudioSegment("For a specific example, if we are here, we already know that Red can just win by making 4 in a row. There is no point in searching any other branches than this one. It would be a waste of our time."));
    c4.inject_audio_and_render(AudioSegment("One last trick that we can take advantage of, is we can guide our search heuristically. If we can make good guesses programmatically about which branches look strong, then this branch-pruning method becomes even more potent."));
    c4.inject_audio_and_render(AudioSegment("Using these strategies, Connect 4 was solved. Thanks to computer scientists such as James Dow Allen, Victor Allis, and John Tromp, we know that Red Wins, and only does so by starting with a disk in the center column."));
    c4.inject_audio_and_render(AudioSegment("One space off to either side and it's a tie, and starting on the edge or one in from the edge, it's Yellow to win."));
    c4.inject_audio_and_render(AudioSegment("But why?"));
    c4.inject_audio_and_render(AudioSegment("This is great and all, but it provides absolutely no information into why, or how, this is the case. "));
    c4.inject_audio_and_render(AudioSegment("It was computed by means of a brute force search, and contributes little to the theoretical analysis of the game."));
    c4.inject_audio_and_render(AudioSegment("All of the strategy, and actual understanding of the game, cannot be communicated by an immense tree of colored nodes. "));
    c4.inject_audio_and_render(AudioSegment("Is there any way to rigorously unite the wisdom and intuition accrued by countless human experts with the output of this enormous algorithm?"));
    c4.inject_audio_and_render(AudioSegment("We can at least do better."));
    c4.inject_audio_and_render(AudioSegment("Before we can truly demystify Connect 4, I need to explain different types of solutions."));
    c4.inject_audio_and_render(AudioSegment("What we talked about before, where I labeled every node in the graph with a color indicating who wins- that is an example of a strong solution."));
    c4.inject_audio_and_render(AudioSegment("We know, for ANY starting position, who is going to win the game."));
    c4.inject_audio_and_render(AudioSegment("But we also mentioned weak solutions. A weak solution is where you know _just enough_ of the game graph to be able to guarantee a win."));
    c4.inject_audio_and_render(AudioSegment("When we were cutting off branches that didn't need to be searched, that was an example of a weak solution. We don't need to memorize stupid variants that we would never even get ourselves into, would we?"));
    c4.inject_audio_and_render(AudioSegment("A strong solution tells us everything that could be known, but weak solutions have a lot more room for creative expression."));
    c4.inject_audio_and_render(AudioSegment("Can we immensely reduce the amount of information required to 'know' a weak solution?"));
    c4.inject_audio_and_render(AudioSegment("Is there a small subset of opening lines that you need to know?"));
    c4.inject_audio_and_render(AudioSegment("Are there any particularly clever ways of representing those weak solutions? We can represent them graph-theoretically, but perhaps they can be expressed by other means?"));
    c4.inject_audio_and_render(AudioSegment("These questions are hard to answer."));
    c4.inject_audio_and_render(AudioSegment("Perhaps that's because they require an intersectional understanding of computer science and a technical understanding of emergent complexities of connect 4 play."));
    c4.inject_audio_and_render(AudioSegment("As far as I can tell, no one has seriously taken up the task of answering them... until now!"));
    c4.inject_audio_and_render(AudioSegment("The answer to all of these questions turns out to be an astonishing yes."));
    c4.inject_audio_and_render(AudioSegment("There ARE extremely low information-density weak solutions with clever representations using Claimeven which we talked about last time."));
    c4.inject_audio_and_render(AudioSegment("We CAN collapse the opening down to a reduced set of irreducibly complex openings."));
    c4.inject_audio_and_render(AudioSegment("There IS a notation by which we can concisely convey the weak solution of sufficiently developed positions."));
    c4.inject_audio_and_render(AudioSegment("What's the point of all this? Through enormous amounts of compute, we were capable of taming connect 4 computationally."));
    c4.inject_audio_and_render(AudioSegment("But the solution that was found is a completely abstract object."));
    c4.inject_audio_and_render(AudioSegment("You can't realistically visualize it. You can't realistically memorize it."));
    c4.inject_audio_and_render(AudioSegment("It may be a landmark in computer science to be able to develop the tools to perform such a task, and I don't mean to take away from the importance of _this result_ (pointing at the opening board)."));
    c4.inject_audio_and_render(AudioSegment("But, the search that was performed yielded a completely intangible theoretical object which we can only get a faint sense for by querying for who wins, one by one, position by position."));
    c4.inject_audio_and_render(AudioSegment("But what is it's form? What is its structure?"));
    c4.inject_audio_and_render(AudioSegment("My goal is to throw all the tricks in the book at this monster and show you what it looks like up close."));
    c4.inject_audio_and_render(AudioSegment("Now, I may be biased, but I think the next few videos are going to be _really_ cool. We're going to take connect four as a case study in the nature of emergent complexity itself. Stay tuned. This has been 2swap."));

    
}
