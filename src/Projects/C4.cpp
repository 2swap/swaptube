#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Media/WhitePaperScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Connect4/C4Scene.cpp"
#include "../Scenes/Connect4/C4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void ideas() {
    CompositeScene cs;
    Graph g;
    string variation = "444";
    shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(&g, false, variation, TRIM_STEADY_STATES);
    shared_ptr<LatexScene> ls_opening = make_shared<LatexScene>("\\text{Opening: "+variation+"}", 1, .2, .1);
    shared_ptr<LatexScene> ls_size = make_shared<LatexScene>("\\text{Node count: "+to_string(g.size())+"}", 1, .2, .1);
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>(variation, .2, .4);
    cs.add_scene(gs, "gs");
    cs.add_scene(ls_opening, "ls_opening", .1, .05);
    cs.add_scene(ls_size, "ls_size", .1, .12);
    cs.add_scene(c4s, "c4s", .1, .26);
    //ValidateC4Graph(g);

    StateSet state{
        {"q1", "{t} .1 * cos"},
        {"qi", "0"},
        {"qj", "{t} .1 * sin"},
        {"qk", "0"},
        {"surfaces_opacity", "0"},
        {"points_opacity", "0"},
        {"physics_multiplier", "30"},
    };
    gs->manager.set(state);

    cs.stage_macroblock(FileBlock("This video isn't about connect four. It's not entirely about computer science, either."), 1);
    cs.stage_macroblock(FileBlock("It's about systems' ability to yield complexity that cannot be expressed in simpler terms."), 1);
    cs.stage_macroblock(FileBlock("It's about our language, and its insufficiency to perfectly describe the world around us."), 1);
    cs.stage_macroblock(FileBlock("It's about emergent behavior- behavior not baked into the 'rules of the game', but arising from them."), 1);
    cs.stage_macroblock(FileBlock("Connect 4 isn't special- the world as we know it contains a myriad of emergent objects built on a bedrock of simple rules."), 1);
    cs.stage_macroblock(FileBlock("It may contain systems like double pendulums, which are fundamentally unpredictable despite being deterministic."), 1);
    cs.stage_macroblock(FileBlock("But that doesn't stop us from navigating and discussing most aspects of the world around us with our finite vocabulary, finite mathematical symbols, or you name it."), 1);
    cs.stage_macroblock(FileBlock("Connect 4 shows us a glimpse of that same emergent substance, distilled down to a system which we can play with and study."), 1);


// Talk about the tree, and how steadystates are sparsely placed on that tree.
    cs.stage_macroblock(FileBlock("Some people have commented that this is all pointless- that connect 4 is solved, it's a closed case."), 1);
    cs.stage_macroblock(FileBlock("And, speaking as someone who has independently solved connect 4 using a novel method,"), 1);
    cs.stage_macroblock(FileBlock("this is dead wrong."), 1);
    cs.stage_macroblock(FileBlock("Not only is there an infinitude of technical details of this simple game that we don't have answers to, and plenty that in my opinion we may never have answers to,"), 1);
    cs.stage_macroblock(FileBlock("but all of them are just reverberations of that one pesky bedrock problem that seemingly lies at the bottom of everything:"), 1);
    cs.stage_macroblock(FileBlock("What is the nature of emergent systems?"), 1);
    cs.stage_macroblock(FileBlock("How can it be that a game with rules simple enough to teach to schoolchildren took mathematicians until the mid 1980s to tell us who should win"), 1);
    cs.stage_macroblock(FileBlock("How can it be that such a game gives rise to all of these bizarre behaviors that demand an entire youtube explainer series along with esoteric German or Japanese words like Zugzwang and Miai to describe them?"), 1);
    cs.stage_macroblock(FileBlock("The rules are simple! Why is there any mystery here???"), 1);
    cs.stage_macroblock(FileBlock("Such is the problem of mathematics."), 1);
    cs.stage_macroblock(FileBlock("We can come up with a set of axioms, but that doesn't mean we know their results."), 1);
    cs.stage_macroblock(FileBlock("After all, the definition of a prime number can be stated concisely in one sentence,"), 1);
    cs.stage_macroblock(FileBlock("but to this day, their distribution is shrouded in mystery."), 1);
    cs.stage_macroblock(FileBlock("Even if we know the forces that guide particle interactions,"), 1);
    cs.stage_macroblock(FileBlock("that doesn't make it any more self-evident that an atom with 43 protons has no stable isotopes,"), 1);
    cs.stage_macroblock(FileBlock("or that this protein is particularly easy at adjoining to this thing which provides biological systems with an easy way to store and recover chemical energy (ATP)"), 1);
    cs.stage_macroblock(FileBlock("Simple rules do not imply obvious behavior."), 1);
    cs.stage_macroblock(FileBlock("Connect 4 provides a very unique combination of properties."), 1);
    cs.stage_macroblock(FileBlock("I hope I have driven home throughout this series that, just like Chess, just like any other sufficiently expressive system, connect 4 yields worlds of complexity beyond the mere language of its axioms."), 1);
    cs.stage_macroblock(FileBlock("Yet, somehow, it is just small enough that I can throw compute at the problem, and make formal mathematical claims that I can back up with actual data."), 1);
    cs.stage_macroblock(FileBlock("And that I can display to you in full!"), 1);
    cs.stage_macroblock(FileBlock("It carries within it that core kernel of emergence, yet on the 7x6 board size, remains just within reach,"), 1);
    cs.stage_macroblock(FileBlock("presenting the mathematician with a unique opportunity for just a glance of that elusive emergent substance."), 1);
    cs.stage_macroblock(FileBlock("This has been 2swap."), 1);
}

void intro(CompositeScene& cs) {
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    shared_ptr<PngScene> sky = make_shared<PngScene>("Sky");
    shared_ptr<PngScene> god1 = make_shared<PngScene>("God1", .3, .6);
    shared_ptr<PngScene> god2 = make_shared<PngScene>("God2", .4, .8);

    cs.stage_macroblock(FileBlock("Suppose two omniscient gods play a game of Connect 4."), 2);
    cs.add_scene_fade_in(MICRO, sky, "sky");
    cs.add_scene_fade_in(MICRO, god1, "god1");
    cs.add_scene_fade_in(MICRO, god2, "god2");
    StateSet floating_gods{
        {"god1.x", "0.2"},
        {"god1.y", "{t} sin .1 * .5 +"},
        {"god2.x", "0.8"},
        {"god2.y", "{t} cos .1 * .5 +"},
    };
    cs.manager.set(floating_gods);
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0.2);
    cs.add_scene_fade_in(MICRO, c4s, "c4");
    cs.move_to_front("god1");
    cs.move_to_front("god2");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Players take turns dropping discs in the columns,"), 1);
    c4s->play("433335526245");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and the first to make a line of 4 wins."), 3);
    c4s->play("4");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "1");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "0");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.5), FileBlock("What would happen?")), 1);
    c4s->undo(100);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("God 1, playing red, plays the first piece in the center column."), 2);
    StateSet god1_opacity = cs.manager.transition(MICRO, "god1.opacity", "1");
    cs.render_microblock();
    c4s->play("4");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("God 2, playing yellow, promptly resigns."), 3);
    cs.manager.transition(MICRO, god1_opacity);
    cs.manager.transition(MICRO, "god2.opacity", "1");
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("god2");
    god2 = make_shared<PngScene>("God2_resign", .4, .8);
    cs.add_scene(god2, "god2");
    cs.manager.set(floating_gods);
    cs.render_microblock();

    Graph g;
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(&g, false, "", LEFTMOST_LOWEST_2);
    c4gs->manager.set({{"dimensions", "2"}, {"physics_multiplier", "<desired_nodes> 100 min"}});
    cs.fade_all_subscenes(MICRO, 0);
    cs.add_scene_fade_in(MICRO, c4gs, "c4gs");
    cs.stage_macroblock(FileBlock("You see, after analyzing every possible variation of every opening, God 2 realized there was no way of stopping God 1 from making a red line of 4."), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4gs");

    cs.stage_macroblock(FileBlock("This was first discovered by computer scientists in 1988."), 2);
    shared_ptr<PngScene> JamesDowAllen = make_shared<PngScene>("JamesDowAllen", .4, .8);
    cs.add_scene(JamesDowAllen, "JDA", .25, 1.45);
    cs.slide_subscene(MICRO, "JDA", 0, -1);
    shared_ptr<LatexScene> ls_jda = make_shared<LatexScene>("\\text{James Dow Allen}", 1, .45, .3);
    cs.add_scene_fade_in(MICRO, ls_jda, "ls_jda", .25, .9);
    cs.render_microblock();
    shared_ptr<PngScene> VictorAllis = make_shared<PngScene>("VictorAllis", .4, .8);
    cs.add_scene(VictorAllis, "VA", .75, -.55);
    cs.slide_subscene(MICRO, "VA", 0, 1);
    shared_ptr<LatexScene> ls_va = make_shared<LatexScene>("\\text{Victor Allis}", 1, .45, .3);
    cs.add_scene_fade_in(MICRO, ls_va, "ls_va", .75, .9);
    cs.render_microblock();

    cs.fade_all_subscenes_except(MICRO, "c4gs", 0);
    cs.stage_macroblock(FileBlock("They used strategies similar to the one God 2 used:"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4gs");

    cs.stage_macroblock(FileBlock("they wrote computer programs to search all possible variations,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("showing that player 1 will win, if they play perfectly."), 1);
    cs.render_microblock();

    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", 4);
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.add_scene(wps, "wps");
    cs.stage_macroblock(FileBlock("Now, don't get me wrong, this was wonderful work by the computer scientists of the day,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But it kind of leaves you wanting."), 1);
    wps->manager.transition(MICRO, "completion", "0", true);
    cs.render_microblock();
    cs.remove_subscene("wps");

    cs.stage_macroblock(FileBlock("Ok, so player 1 wins, but _why_?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A computer might be able to iterate over billions of positions to check this result,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But what's left for us humans?"), 1);
    shared_ptr<PngScene> human1 = make_shared<PngScene>("Thinker1", .3, .6);
    shared_ptr<PngScene> human2 = make_shared<PngScene>("Thinker2", .3, .6);
    cs.add_scene(human1, "human1", .15, 1.7);
    cs.add_scene(human2, "human2", .85, 1.7);
    cs.slide_subscene(MICRO, "human1", 0, -1);
    cs.slide_subscene(MICRO, "human2", 0, -1);
    cs.render_microblock();

    cs.add_scene_fade_in(MICRO, c4s, "c4");
    cs.fade_subscene(MICRO, "c4gs", 0);
    cs.stage_macroblock(FileBlock("What are the best openings as player 2, to make it as hard as possible for player 1?"), 5);
    cs.render_microblock();
    cs.remove_subscene("c4gs");
    c4s->play("4");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();
    c4s->play("3");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Is there some change of perspective that shows how mere mortals could win against god 2?"), 1);
    cs.slide_subscene(MICRO, "human2", 0, 1);
    god2 = make_shared<PngScene>("God2", .4, .8);
    cs.add_scene_fade_in(MICRO, god2, "god2", .75, -.5);
    cs.manager.transition(MICRO, floating_gods);
    cs.render_microblock();
    cs.remove_subscene("human2");

    cs.stage_macroblock(FileBlock("These questions went unanswered... until now."), 1);
    cs.render_microblock();
}

void build_graph(CompositeScene& cs) {
    cs.fade_all_subscenes(MICRO, 0);

    Graph g;
    shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(&g, false, "", TRIM_STEADY_STATES);
    cs.add_scene(gs, "gs");

    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"decay",".5"},
        {"surfaces_opacity","0"},
        {"points_opacity","0"},
        {"mirror_force",".005"},
        {"desired_nodes", "10000 4 <time_since_graph_init> ^ 20 * min"},
        {"dimensions", "3"},
        {"physics_multiplier", "<desired_nodes> sqrt"},
        {"flip_by_symmetry", "0"},
    });
    gs->manager.transition(MICRO, "decay", ".4");

    cs.stage_macroblock(FileBlock("Because today, I'm going to show you connect 4 through the eyes of God 1."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 2);

    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{ALL {\\scriptsize IT} {\\small TAKES}}", 1, .6, .3);
    cs.add_scene(ls, "ls", .3, .15);
    cs.render_microblock();

    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("\\text{{\\scriptsize TO PLAY} PERFECTLY}", 1, .8, .4);
    cs.add_scene(ls2, "ls2", .6, .8);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    gs->manager.transition(MICRO, "d", "-1");
    gs->manager.transition(MICRO, "physics_multiplier", "0");
    cs.fade_all_subscenes_except(MICRO, "gs", 0);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

void explanation(CompositeScene& cs) {
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MACRO, c4s, "c4s");
    cs.stage_macroblock(FileBlock("Suppose you're going to play as red, and you want a strategy to play perfectly."), 3);
    cs.render_microblock();
    string starting_variation = "4443674433";
    c4s->play(starting_variation);
    cs.render_microblock();
    cs.render_microblock();

    c4s->set_fast_mode(true);
    int vars_to_read = 5;
    int vars_depth = 20;
    cs.stage_macroblock(FileBlock("What options are there, besides running brute force search in your head?"), vars_to_read * 2);
    for(int i = 0; i < vars_to_read; i++) {
        // String of random numbers in [1,7], length vars_depth
        string str = "";
        C4Board b(starting_variation);
        for(int j = 0; j < vars_depth; j++) {
            int move = rand() % 7 + 1;
            cout << move << flush;
            if(!b.is_legal(move)) {
                cout << "x" << flush;
                j--;
                continue;
            }
            C4Board child(b);
            child.play_piece(move);
            if(child.who_won() != INCOMPLETE) {
                cout << "w" << flush;
                j--;
                continue;
            }
            b.play_piece(move);
            str += to_string(move);
        }
        c4s->play(str);
        cout << str << endl;
        cs.render_microblock();
        c4s->undo(vars_depth);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("Instead you could memorize every branch upfront."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That way you can just recall the right response during the tournament."), 1);
    cs.render_microblock();

    shared_ptr<PngScene> terabytes = make_shared<PngScene>("15TB", .6, .6);
    cs.stage_macroblock(FileBlock("Well, you'll need to memorize 15 terabytes of connect 4 positions, as this poor soul found out."), 3);
    double dividing_line = 0.15;
    terabytes->manager.set("crop_bottom", to_string(dividing_line));
    cs.render_microblock();
    terabytes->manager.transition(MICRO, {
            {"crop_top", to_string(1-dividing_line)},
            {"crop_bottom", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    return;

    Graph g;
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(&g, false, "", FULL);

    cs.add_scene(c4gs, "c4gs");
    cs.stage_macroblock(FileBlock("These two strategies- brute force search and upfront memorization- involve the same tree of positions."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The difference is that the former makes you think a lot, and the latter makes you remember a lot."), 2);
    shared_ptr<PngScene> cpu = make_shared<PngScene>("cpu", .4, .4);
    cs.add_scene_fade_in(MICRO, cpu, "cpu", .25, .5);
    cs.render_microblock();
    shared_ptr<PngScene> hdd = make_shared<PngScene>("hdd", .4, .4);
    cs.add_scene_fade_in(MICRO, hdd, "hdd", .75, .5);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<RealFunctionScene> rfs= make_shared<RealFunctionScene>();
    rfs->manager.set("zero_crosshair_opacity", "1");
    rfs->manager.set({{"center_x", "3"}, {"center_y", "<center_x>"}, {"zoom", "-1.5"}});
    cs.add_scene_fade_in(MICRO, rfs, "rfs");
    cs.render_microblock();
    cs.remove_all_subscenes_except("rfs");

    cs.stage_macroblock(FileBlock("Plotting the amount of variations that we would need to search with brute force after some amount of moves,"), 1);
    rfs->manager.transition(MICRO, "function0", "(a) exp");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("it looks like this."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And plotting the positions you'd need to memorize to get to the nth move,"), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function1", "<center_x> 2 * (a) - exp");

    cs.stage_macroblock(FileBlock("we see the opposite curve."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This suggests a hybrid approach-"), 1);
    cs.render_microblock();

    string hybrid = "(a) exp <center_x> 2 * (a) - exp min";

    rfs->manager.transition(MICRO, "function0", hybrid);
    cs.stage_macroblock(FileBlock("Memorizing all positions only up to the midgame,"), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function1", hybrid);
    cs.stage_macroblock(FileBlock("and using search after the middle of the game,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we can balance compute and memory to avoid these exponential explosions."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And it makes sense- this is exactly what chess players do in practice."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("They memorize openings, and practice reading after the position is already developed."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("One of those early solvers of connect 4, Victor Allis, published just that- a table of 500,000 openings, and a computer algorithm which could solve endgames in a few seconds."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now I don't know about you, but I can't memorize half a million positions."), 1);
}

void render_video() {
    CompositeScene cs;
    //intro(cs);
    //build_graph(cs);
    explanation(cs);
}
