#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Media/Mp4Scene.cpp"
#include "../Scenes/Media/WhitePaperScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Connect4/C4Scene.cpp"
#include "../Scenes/Media/SvgScene.cpp"
#include "../Scenes/Connect4/C4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void ideas() {
    CompositeScene cs;
    shared_ptr<Graph> g = make_shared<Graph>();
    string variation = "444";
    shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(g, false, variation, TRIM_STEADY_STATES);
    shared_ptr<LatexScene> ls_opening = make_shared<LatexScene>("\\text{Opening: "+variation+"}", 1, .2, .1);
    shared_ptr<LatexScene> ls_size = make_shared<LatexScene>("\\text{Node count: "+to_string(g->size())+"}", 1, .2, .1);
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

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(g, false, "", LEFTMOST_LOWEST_2);
    c4gs->manager.set("desired_nodes", "100 1.1 <time_since_graph_init> ^ 1 - * 1000000 min");
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

    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", vector<int>{1, 7, 10, 82});
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.add_scene(wps, "wps");
    cs.stage_macroblock(FileBlock("Now, don't get me wrong, this was wonderful work by the computer scientists of the day,"), 1);
    cs.render_microblock();
    cs.remove_subscene("c4gs");

    cs.stage_macroblock(FileBlock("But it kind of leaves you wanting."), 1);
    wps->manager.transition(MICRO, "completion", "0", true);
    cs.render_microblock();
    cs.remove_subscene("wps");

    c4s = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MICRO, c4s, "c4");
    cs.stage_macroblock(FileBlock("Ok, so player 1 wins, but _why_?"), 2);
    cs.render_microblock();
    c4s->play("4");
    cs.render_microblock();

    god2 = make_shared<PngScene>("God2", .4, .8);
    cs.add_scene(god2, "god2_notice", .75, -.5);
    cs.manager.transition(MICRO, floating_gods);
    cs.stage_macroblock(FileBlock("A computer, or a god, can iterate over billions of positions to check this result,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But what's left for us humans?"), 1);
    cs.fade_subscene(MICRO, "god2", 0);
    shared_ptr<PngScene> human1 = make_shared<PngScene>("Thinker1", .3, .6);
    shared_ptr<PngScene> human2 = make_shared<PngScene>("Thinker2", .3, .6);
    cs.add_scene(human1, "human1", .15, 1.7);
    cs.add_scene(human2, "human2", .85, 1.7);
    cs.slide_subscene(MICRO, "human1", 0, -1);
    cs.slide_subscene(MICRO, "human2", 0, -1);
    cs.render_microblock();
    cs.remove_subscene("god2");

    cs.stage_macroblock(FileBlock("After player 1 plays in the center, what player 2 openings make it as hard as possible for player 1?"), 2);
    cs.render_microblock();
    shared_ptr<PngScene> human2_trick = make_shared<PngScene>("Thinker2_trick", .3, .6);
    cs.add_scene_fade_in(MICRO, human2_trick, "human2_trick", .85, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("To confuse red, should player 2 respond in the center, or one over? What about all the way on the side?"), 8);
    cs.fade_subscene(MICRO, "human2_trick", 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("human2_trick");
    c4s->play("4");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();
    c4s->play("3");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();
    c4s->play("7");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Is there some change of perspective that shows how us mortals could win against god 2?"), 1);
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

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(g, false, "", TRIM_STEADY_STATES);
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
        {"desired_nodes", "10000 3 <time_since_graph_init> ^ 20 * min"},
        {"dimensions", "3"},
        {"physics_multiplier", "<desired_nodes> sqrt"},
        {"flip_by_symmetry", "0"},
    });
    gs->manager.transition(MICRO, "decay", ".4");

    cs.stage_macroblock(FileBlock("Because today, I'm going to show you connect 4 through the eyes of God 1."), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Because I found a profoundly simpler way to play connect 4. All it takes to play perfectly is this graph."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 1);
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
        C4Board b(FULL, starting_variation);
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

    cs.stage_macroblock(FileBlock("That way you can just recall the right response during the game."), 1);
    cs.render_microblock();

    shared_ptr<PngScene> terabytes = make_shared<PngScene>("15TB", .6, .6);
    cs.add_scene_fade_in(MICRO, terabytes, "terabytes");
    cs.stage_macroblock(FileBlock("Well, you'll need to memorize 15 terabytes of connect 4 positions, as this poor soul found out."), 3);
    cs.render_microblock();
    terabytes->manager.transition(MICRO, {
            {"crop_top", ".87"},
            {"crop_bottom", ".07"},
            {"crop_right", ".355"},
            {"crop_left", ".05"},
            {"w", ".8"},
            {"h", ".8"},
    });
    cs.render_microblock();
    cs.render_microblock();

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(g, false, "", FULL);

    cs.fade_all_subscenes(MICRO, 0);
    cs.add_scene(c4gs, "c4gs");
    cs.stage_macroblock(FileBlock("These two strategies- brute force search and upfront memorization- involve the same tree of positions."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The difference is that the former makes you think a lot, and the latter makes you remember a lot."), 2);
    cs.fade_subscene(MICRO, "c4gs", 0);
    shared_ptr<PngScene> cpu = make_shared<PngScene>("cpu", .4, .4);
    cs.add_scene_fade_in(MICRO, cpu, "cpu", .25, .5);
    cs.render_microblock();
    cs.remove_subscene("c4gs");
    shared_ptr<PngScene> hdd = make_shared<PngScene>("hdd", .4, .4);
    cs.add_scene_fade_in(MICRO, hdd, "hdd", .75, .5);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<RealFunctionScene> rfs= make_shared<RealFunctionScene>();
    rfs->manager.set("ticks_opacity", "0");
    rfs->manager.set("zero_crosshair_opacity", "1");
    rfs->manager.set("function0_opacity", "1");
    rfs->manager.set("function1_opacity", "1");
    rfs->manager.set("function0_right", "0");
    rfs->manager.set("function0_left", "0");
    rfs->manager.set("function1_right", "0");
    rfs->manager.set("function1_left", "0");
    string function0 = "1.4 (a) <center_x> - ^";
    string function1 = "1.4 -1 (a) * <center_x> + ^";
    rfs->manager.set("function0", function0);
    rfs->manager.set("function1", function1);
    rfs->manager.set({{"center_x", "8"}, {"center_y", "5"}, {"zoom", "-1"}});
    cs.add_scene_fade_in(MICRO, rfs, "rfs");
    cs.render_microblock();
    cs.remove_all_subscenes_except("rfs");

    cs.stage_macroblock(FileBlock("Plotting the amount of positions up to n moves, you get a curve like this."), 2);
    rfs->manager.transition(MICRO, "function0_right", "20");
    cs.render_microblock();

    cs.fade_subscene(MICRO, "rfs", 0.3);
    shared_ptr<C4Scene> quick_display = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MICRO, quick_display, "quick_display");
    quick_display->set_fast_mode(true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There's 7 options for the first move,"), 8);
    for(int i = 1; i <= 7; i++) {
        quick_display->undo(1);
        quick_display->play(to_string(i));
        cs.render_microblock();
    }
    quick_display->undo(1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but after 5 moves there's already over 5000 possibilities."), 1);
    for(int i = 0; i < 1000; i++) {
        string random_sequence = "";
        for(int j = 0; j < 5; j++){
            char ch = (char)(rand() % 7 + '1');
            random_sequence += ch;
        }
        quick_display->play(random_sequence);
        quick_display->undo(5);
    }
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Classic exponential growth!"), 1);
    cs.slide_subscene(MICRO, "quick_display", 0, 1);
    cs.fade_subscene(MICRO, "rfs", 1);
    cs.render_microblock();
    cs.remove_subscene("quick_display");

    cs.stage_macroblock(FileBlock("Now, after n moves have already transpired, if we plot the amount of variations that need to be searched to brute-force solve the rest of the game..."), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function1_right", "20");

    cs.stage_macroblock(FileBlock("we see an opposite curve. It's easier to work out a game that's almost over than one which has just started."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This suggests a hybrid approach-"), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function0_right", "<center_x>");
    cs.stage_macroblock(FileBlock("Memorizing all positions only up to the midgame,"), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function1_left", "<center_x>");
    cs.stage_macroblock(FileBlock("and using search after the middle of the game,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we can balance compute and memory to avoid these exponential explosions."), 1);
    cs.render_microblock();

    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<LatexScene> ls_memo = make_shared<LatexScene>("\\text{Common opening worth memorizing}", 1, .8, .3);
    shared_ptr<LatexScene> ls_comp = make_shared<LatexScene>("\\text{Rare variation, switch to reading ahead}", 1, .8, .3);
    cs.add_scene_fade_in(MICRO, ls_memo, "ls_memo", .5, .1);
    cs.add_scene(ls_comp, "ls_comp", .5, .1);
    cs.manager.set("ls_comp.opacity", "0");
    shared_ptr<Mp4Scene> chess_clip = make_shared<Mp4Scene>(vector<string>{"chess"}, 3, .6, .6);
    cs.add_scene_fade_in(MICRO, chess_clip, "chess_clip");
    StateSet frame = chess_clip->manager.set("current_frame", "0");
    cs.stage_macroblock(FileBlock("This is why chess players memorize openings, and start reading after the position has developed beyond their memory."), 5);
    cs.render_microblock();
    cs.manager.begin_timer("MP4_Frame");
    cs.manager.set(frame);
    chess_clip->manager.set("current_frame", "[current_frame]");
    cs.manager.set({
        {"ls_memo.x", ".5 <current_frame> 230 - 20 / max"},
        {"ls_comp.x", ".5 <current_frame> 250 - 20 / min"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", vector<int>{1, 7, 10, 82});
    cs.add_scene(wps, "wps");
    cs.stage_macroblock(FileBlock("Victor Allis published just that in 1988: a table of 500,000 openings, and a computer algorithm to solve endgames in a few seconds."), 12);
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.set("which_page", "82");
    wps->manager.transition(MICRO, "page_focus", "1");
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".253"}, {"crop_bottom", ".707"}});
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".398"}, {"crop_bottom", ".572"}, {"crop_left", ".1"}, {"crop_right", ".5"}});
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now I don't know about you, but I can't memorize half a million positions."), 2);
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});
    cs.render_microblock();
    cs.render_microblock();
}

void trees(CompositeScene& cs) {
    cs.stage_macroblock(FileBlock("We can improve on this- there's one key insight that the prior approaches don't make use of."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Before I tell you, let me explain the various types of trees that I've been showing in the background."), 1);
    cs.render_microblock();

    shared_ptr<Graph> g1 = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(g1, false, "", FULL);
    c4gs->manager.set("points_opacity", "0");
    c4gs->manager.set("points_radius_multiplier", "0.5");
    c4gs->manager.set("desired_nodes", "5678 1.5 <time_since_graph_init> ^ 200 * 199 - min");
    cs.add_scene(c4gs, "c4gs");
    cs.stage_macroblock(FileBlock("If we add every possible connect 4 position to a tree, we get this gigantic thing."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This full game tree up to move 5, taken as a guide on where to play, is called a 'strong solution'."), 2);
    cs.render_microblock();
    shared_ptr<LatexScene> ls_strong = make_shared<LatexScene>("\\text{Strong Solution {\\tiny (Depth 5)}}", 1, .5, .2);
    cs.add_scene_fade_in(MICRO, ls_strong, "ls_strong", .5, .1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ls_strong", 0);
    cs.stage_macroblock(FileBlock("However, this tree includes nodes where player 1, red, has already blundered."), 1);
    cs.render_microblock();
    cs.remove_subscene("ls_strong");
    
    cs.stage_macroblock(FileBlock("We don't want to memorize those, since we'd never play them."), 1);
    cs.render_microblock();

    shared_ptr<Graph> g2 = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs_union = make_shared<C4GraphScene>(g2, false, "", UNION_WEAK);
    g2->expand(5677); // At least as deep as the full tree
    g2->make_bidirectional();

    // Remove all nodes in the g1 which are not in g2
    vector<double> to_delete;
    auto it = g1->nodes.begin();
    for (; it != g1->nodes.end(); ++it) {
        if (g2->nodes.find(it->first) == g2->nodes.end()) {
            cout << "Deleting node: " << it->first << endl;
            to_delete.push_back(it->first);
            cout << "to_delete size: " << to_delete.size() << endl;
        }
    }

    c4gs->manager.set("desired_nodes", "0");
    cs.stage_macroblock(FileBlock("If we trim those blunders out, we get this smaller tree."), to_delete.size());
    for(const double& id : to_delete) {
        g1->remove_node(id);
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can shrink it further though."), 1);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("45");
    cs.add_scene(c4s, "c4s", 1.5, .5);
    cs.slide_subscene(MICRO, "c4s", -.75, 0);
    cs.slide_subscene(MICRO, "c4gs", -.25, 0);
    cs.stage_macroblock(FileBlock("Consider this position."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It's red's move, and there's two winning options."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Red could either go here,"), 2);
    c4s->play("2");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("or here."), 1);
    c4s->play("1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Everything else would be a blunder."), 1);
    cs.slide_subscene(MICRO, "c4s", .75, 0);
    cs.slide_subscene(MICRO, "c4gs", .25, 0);
    cs.render_microblock();
    cs.remove_subscene("c4s");

    it = g1->nodes.begin();
    for (; it != g1->nodes.end(); ++it) {
        it->second.opacity = 0;
        if (it->second.data->representation == "452" || it->second.data->representation == "451") {
            it->second.opacity = 1;
            it->second.size = 15;
        }
    }
    cs.stage_macroblock(FileBlock("Those two nodes are in our tree."), 1);
    c4gs->manager.transition(MICRO, "points_opacity", "1");
    cs.render_microblock();

    shared_ptr<Graph> g3 = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs_weak = make_shared<C4GraphScene>(g3, false, "", RANDOM_WEAK);
    g3->expand(5677); // At least as deep as the full tree
    g3->make_bidirectional();

    to_delete.clear();
    it = g1->nodes.begin();
    cout << "g1 size: " << g1->nodes.size() << endl;
    for (; it != g1->nodes.end(); ++it) {
        if (g3->nodes.find(it->first) == g3->nodes.end()) {
            if(it->second.data->representation == "452" || it->second.data->representation == "451") {
                to_delete.push_back(it->first);
            }
        }
    }

    cs.stage_macroblock(FileBlock("But why memorize both options, instead of just memorizing the one we like better?"), 2);
    cs.render_microblock();
    for(const double& id : to_delete) {
        g1->remove_node(id);
    }
    cs.render_microblock();

    // Delete all nodes not in c4gs_weak
    to_delete.clear();
    it = g1->nodes.begin();
    for (; it != g1->nodes.end(); ++it) {
        it->second.size = 1;
        if(g3->nodes.find(it->first) == g3->nodes.end()) {
            to_delete.push_back(it->first);
        }
    }

    cs.stage_macroblock(FileBlock("Let's delete each duplicate option."), to_delete.size() + 1);
    cs.render_microblock();
    for(const double& id : to_delete) {
        g1->remove_node(id);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("This tree is sufficient- we can still win by memorizing it. We've just cut out all the stupid variations that we don't need."), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls_weak = make_shared<LatexScene>("\\text{Weak Solution {\\tiny (Depth 5)}}", 1, .5, .2);
    cs.add_scene_fade_in(MICRO, ls_weak, "ls_weak", .5, .1);
    cs.stage_macroblock(FileBlock("This is a 'weak solution'."), 1);
    cs.render_microblock();

    cs.fade_all_subscenes(MICRO, 0);
    cs.stage_macroblock(FileBlock("Now, let me provide some inspiration for the key insight."), 1);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

int find_downstream_size(shared_ptr<Graph> g, const string& variation) {
    C4Board b(FULL, variation);
    double current_id = b.get_hash();

    // Iterate over all graph nodes, and find the "downstream size" of a variation.
    // In other words, how many nodes are reachable from this node.
    set<double> visited;
    function<void(double)> dfs = [&](double current_id) {
        if (visited.find(current_id) != visited.end()) {
            return;
        }
        visited.insert(current_id);
        Node& current_node = g->nodes.find(current_id)->second;
        const string& current_rep = current_node.data->representation;
        for (const Edge& neighbor_edge : current_node.neighbors) {
            double neighbor_id = neighbor_edge.to;
            Node& neighbor_node = g->nodes.find(neighbor_id)->second;
            const string& neighbor_rep = neighbor_node.data->representation;
            // Only traverse to neighbors that are one move ahead in the game
            if (neighbor_rep.length() == current_rep.length() + 1)
                dfs(neighbor_id);
        }
    };
    dfs(current_id);
    return visited.size();
}

void patterned(CompositeScene& cs) {
    string variation = "43636335555665773563";
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MICRO, c4s, "c4s");
    cs.stage_macroblock(FileBlock("Check out this game. It's almost over, with Red to move."), 3);
    cs.render_microblock();
    c4s->play(variation);
    cs.render_microblock();
    cs.render_microblock();

    c4s->manager.transition(MICRO, {{"w", ".3"}, {"h", ".3"}});
    cs.slide_subscene(MICRO, "c4s", 0, -.35);

    cs.stage_macroblock(FileBlock("Now what do you think the weak solution looks like for this endgame?"), 1);
    cs.render_microblock();

    shared_ptr<Graph> g_weak_1 = make_shared<Graph>();
    shared_ptr<Graph> g_weak_2 = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs_weak_1 = make_shared<C4GraphScene>(g_weak_1, false, variation, RIGHTMOST_WEAK);
    shared_ptr<C4GraphScene> c4gs_weak_2 = make_shared<C4GraphScene>(g_weak_2, false, variation, SIMPLE_WEAK);
    c4gs_weak_1->manager.set("physics_multiplier", "50");
    c4gs_weak_2->manager.set("physics_multiplier", "50");
    c4gs_weak_2->manager.set("growth_rate", "5");
    cs.stage_macroblock(FileBlock("I'll give you 2 options."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Do you think it looks like this messy tree?"), 1);
    cs.add_scene(c4gs_weak_1, "c4gs_weak_1", .25, .5);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Or do you think that it looks like this nice pillow shape?"), 1);
    cs.add_scene(c4gs_weak_2, "c4gs_weak_2", .75, .5);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This isn't just some trivial question about appearance- if there is a particularly simple weak solution, that means we can compress the information contained within it."), 1);
    cs.render_microblock();

    //TODO pause timer scene
    cs.stage_macroblock(FileBlock("So, make your guess! Which one do you think it is?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The answer is... both of them!"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0.2);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s_compare = make_shared<C4Scene>("");
    cs.stage_macroblock(FileBlock("Remember how I chose between two winning options earlier?"), 5);
    cs.add_scene_fade_in(MICRO, c4s_compare, "c4s_compare");
    cs.render_microblock();
    c4s_compare->play("45");
    cs.render_microblock();
    c4s_compare->play("2");
    cs.render_microblock();
    c4s_compare->undo(1);
    cs.render_microblock();
    c4s_compare->play("1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Up to the choices we make, we can get much different trees!"), 1);
    cs.fade_subscene(MICRO, "c4s_compare", 0);
    cs.fade_subscene(MICRO, "c4gs_weak_2", 1);
    cs.fade_subscene(MICRO, "c4gs_weak_1", 1);
    cs.fade_subscene(MICRO, "c4s", 1);
    cs.render_microblock();
    cs.remove_subscene("c4s_compare");

    c4gs_weak_1->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.slide_subscene(MICRO, "c4gs_weak_1", .25, 0);
    cs.fade_subscene(MICRO, "c4gs_weak_2", 0);
    cs.stage_macroblock(FileBlock("I made this one by always choosing the rightmost winning column."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "c4gs_weak_1", 0);
    cs.fade_subscene(MICRO, "c4gs_weak_2", 1);
    cs.render_microblock();
    cs.remove_subscene("c4gs_weak_1");

    c4gs_weak_2->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.slide_subscene(MICRO, "c4gs_weak_2", -.25, 0);
    cs.stage_macroblock(FileBlock("So how on earth did I make this one?"), 1);
    cs.render_microblock();

    cs.fade_all_subscenes_except(MICRO, "c4s", 0);
    c4s->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.slide_subscene(MICRO, "c4s", 0, .35);
    cs.stage_macroblock(FileBlock("How do we identify these patterny weak solutions?"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4s");

    c4s->set_annotations_from_steadystate(MICRO);
    cs.stage_macroblock(FileBlock("I use what I call a 'steady state diagram'."), 1);
    cs.render_microblock();
    c4s->clear_annotations(MICRO);

    cs.stage_macroblock(FileBlock("This is a particularly simple example. Let's look at a more realistic one."), 3);
    cs.render_microblock();
    c4s->undo(100);
    cs.render_microblock();
    c4s->play("473534");
    cs.render_microblock();
    c4s->set_annotations_from_steadystate(MICRO);

    cs.stage_macroblock(FileBlock("This is what a steady state diagram looks like in a real game."), 1);
    cs.render_microblock();

    cs.slide_subscene(MICRO, "c4s", .2, 0);
    cs.stage_macroblock(FileBlock("The diagram is like a cheat-sheet, telling Red how to make all the right moves from here until the end of the game."), 1);
    cs.render_microblock();

    shared_ptr<SvgScene> rules = make_shared<SvgScene>("steady_state_rules_0", 1, .5, 1);
    cs.add_scene(rules, "rules", .25, .5);
    cs.stage_macroblock(FileBlock("To read it, there's a series of 8 rules."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The rules are organized in a list of priorities."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In this game, it's red's turn. Let's follow the diagram to win the game."), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_1");
    cs.stage_macroblock(FileBlock("Rule 1: Is there a winning move available?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    cs.stage_macroblock(FileBlock("Nope. Move on to rule 2."), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_2");
    cs.stage_macroblock(FileBlock("Is the opponent about to win, such that we need to block it?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    cs.stage_macroblock(FileBlock("Nope. Continue!"), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_3");
    cs.stage_macroblock(FileBlock("Can we play on an exclamation mark?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Yes!"), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    cs.stage_macroblock(FileBlock("That's the move we'll take."), 2);
    c4s->play("3");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Our opponent blocks the threat. What should we do now?"), 1);
    c4s->play("3");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We go through the rules, checking each one, but none of them apply..."), 5);
    rules->begin_svg_transition(MICRO, "steady_state_rules_1");
    cs.render_microblock();
    rules->begin_svg_transition(MICRO, "steady_state_rules_2");
    cs.render_microblock();
    rules->begin_svg_transition(MICRO, "steady_state_rules_3");
    cs.render_microblock();
    rules->begin_svg_transition(MICRO, "steady_state_rules_4");
    cs.render_microblock();
    rules->begin_svg_transition(MICRO, "steady_state_rules_5");
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_6");
    cs.stage_macroblock(FileBlock("until rule 6, which asks if there's a plus sign we can play on."), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Yes!"), 1);
    c4s->play("2");
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    cs.stage_macroblock(FileBlock("Our opponent plays here."), 2);
    cs.render_microblock();
    c4s->play("1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Try and figure out what Red should do next, according to the rules."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_4");
    cs.stage_macroblock(FileBlock("There's only one at sign available now, so Red plays on it."), 2);
    c4s->play("4");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As Red, if you follow this cheat sheet, regardless of what Yellow does, you'll eventually win the game."), 5);
    c4s->play("413357447766661166117755");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "1");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "0");
    cs.render_microblock();

    cs.fade_all_subscenes_except(MICRO, "c4s", 0);
    cs.stage_macroblock(SilenceBlock(2), 1);
    c4s->clear_annotations(MICRO);
    c4s->undo(100);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4s");

    cs.stage_macroblock(FileBlock("These diagrams are effectively a form of compression."), 1);
    c4s->play(variation);
    cs.render_microblock();
    c4s->set_annotations_from_steadystate(MICRO);

    cs.stage_macroblock(FileBlock("By identifying such a simple program to solve a connect 4 position, we can memorize that program instead of some convoluted tree traversed by brute force search."), 1);
    cs.add_scene_fade_in(MICRO, c4gs_weak_1, "c4gs_weak_1", .25, .5);
    cs.render_microblock();

    cs.slide_subscene(MICRO, "c4s", -.2, 0);
    cs.fade_subscene(MICRO, "c4gs_weak_1", 0);
    cs.stage_macroblock(FileBlock("Now, one obvious question is: what is the cheat sheet for the starting position?"), 1);
    c4s->undo(variation.length());
    c4s->clear_annotations(MICRO);
    cs.render_microblock();
    cs.remove_subscene("c4gs_weak_1");

    cs.stage_macroblock(FileBlock("Unfortunately we're not that lucky."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("These diagrams are only good for describing midgames when the path to victory is already somewhat well-defined."), 2);
    cs.render_microblock();
    c4s->play("44444221");
    cs.render_microblock();
    c4s->set_annotations_from_steadystate(MICRO);
}

void trimmed_solution(CompositeScene& cs) {
    cs.stage_macroblock(FileBlock("So we're back to old-reliable: brute-force search."), 2);
    cs.fade_subscene(MICRO, "c4s", 0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<C4GraphScene> weakc4 = make_shared<C4GraphScene>(g, false, "", TRIM_STEADY_STATES);
    if(!FOR_REAL) {
        g->expand(-1);
        g->make_bidirectional();
    }
    cs.add_scene(weakc4, "weakc4");
    weakc4->manager.set("physics_multiplier", "60");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But this time, we can terminate search on positions which have a steady state solution."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And if we furthermore try to find the smallest such graph- that is, the one that most quickly directs you to those easy cheat-sheet positions, you get something like this."), 1);
    cs.render_microblock();

    weakc4->manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "{t} 10 / sin 10 /"}, {"qk", "0"}});
    cs.stage_macroblock(FileBlock("The first thing you might notice is that it's symmetrical."), 1);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s_left = make_shared<C4Scene>("", .5, 1);
    shared_ptr<C4Scene> c4s_right = make_shared<C4Scene>("", .5, 1);
    cs.add_scene(c4s_left, "c4s_left", -.25, .5);
    cs.add_scene(c4s_right, "c4s_right", 1.25, .5);
    cs.slide_subscene(MICRO, "c4s_left", .5, 0);
    cs.slide_subscene(MICRO, "c4s_right", -.5, 0);
    cs.stage_macroblock(FileBlock("This makes sense- If yellow plays on the left versus on the right, we'll respond in a way that is mirrored horizontally."), 4);
    cs.render_microblock();
    c4s_left->play("444");
    c4s_right->play("444");
    cs.render_microblock();
    c4s_left->play("3");
    c4s_right->play("5");
    cs.render_microblock();
    c4s_left->play("6");
    c4s_right->play("2");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There's a few exceptions to this rule. For example, this section only has one asymmetric copy."), 3);
    cs.slide_subscene(MICRO, "c4s_left", -.5, 0);
    cs.slide_subscene(MICRO, "c4s_right", .5, 0);
    cs.render_microblock();
    cs.remove_subscene("c4s_left");
    cs.remove_subscene("c4s_right");

    C4Board asymm_board(FULL, "4444445");
    double hash = asymm_board.get_hash();
    glm::vec4 pos = g->nodes.find(hash)->second.position;
    weakc4->manager.transition(MICRO, {{"x", to_string(pos.x)}, {"y", to_string(pos.y)}, {"z", to_string(pos.z)}, {"d", ".5"}});
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In this spot, the two players made a perfectly symmetrical opening, and Red is forced to pick a side to continue on."), 1);
    cs.render_microblock();

    weakc4->manager.set("desired_nodes", "0");

    list<pair<double, double>> to_merge;
    for(auto& pair : g->nodes) {
        double id = pair.first;
        C4Board b = *(static_cast<C4Board*>(pair.second.data));
        C4Board b_mirror = b.get_mirror_board();
        double mirror_id = b_mirror.get_hash();
        if(g->node_exists(mirror_id) && id < mirror_id) {
            to_merge.push_back({id, mirror_id});
        }
    }
    cs.stage_macroblock(FileBlock("Let's fuse together mirror-symmetric nodes!"), 1);
    weakc4->manager.transition(MICRO, {{"x", "0"}, {"y", "0"}, {"z", "0"}, {"d", "1"}});
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), to_merge.size());
    for(const auto& p : to_merge) {
        g->collapse_two_nodes(p.first, p.second);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("This connect 4 solution has only 4,550 nodes!"), 1);
    cs.render_microblock();


    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", vector<int>{1, 7, 10, 82});
    cs.add_scene(wps, "wps");
    cs.stage_macroblock(FileBlock("Comparing against Victor Allis's solution from the eighties,"), 2);
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.render_microblock();
    wps->manager.set("which_page", "82");
    wps->manager.transition(MICRO, "page_focus", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Allis's opening book was more than a hundred times as large, with half a million nodes."), 4);
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});

    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".253"}, {"crop_bottom", ".707"}});
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Furthermore, unlike the few seconds of search that Allis used,"), 2);
    wps->manager.transition(MICRO, {{"crop_top", ".398"}, {"crop_bottom", ".572"}, {"crop_left", ".1"}, {"crop_right", ".5"}});
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we don't need any search whatsoever since the steady state diagrams instantly tell us what to do in the endgames."), 1);
    cs.fade_subscene(MICRO, "wps", 0);
    cs.render_microblock();
    cs.remove_subscene("wps");

    for(int i = 2; i < 8; i+=2) {
        unordered_map<string, int> length_count;
        cout << "Analyzing downstream sizes for length " << i << " variations." << endl;
        for(auto& pair : g->nodes) {
            Node& node = pair.second;
            string variation = node.data->representation;
            if(variation.length() != i)
                continue;
            int downstream_size = find_downstream_size(g, variation);
            length_count[variation] = downstream_size;
        }
        // print out in sorted order, starting with biggest
        vector<pair<string, int>> sorted_length_count(length_count.begin(), length_count.end());
        sort(sorted_length_count.begin(), sorted_length_count.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second > b.second;
        });
        cout << "Downstream sizes for length " << i << " variations:" << endl;
        for(const auto& p : sorted_length_count) {
            cout << "Variation: " << p.first << ", Downstream Size: " << p.second << endl;
        }
    }
}

void flood_fill_edges_to_highlight(shared_ptr<Graph> g, double start_node_id) {
    // First set all edges to low opacity
    for (auto& pair : g->nodes) {
        Node& node = pair.second;
        for (const Edge& edge : node.neighbors) {
            const_cast<Edge&>(edge).opacity = 0.2;
        }
    }

    set<double> visited;
    function<void(double)> dfs = [&](double current_id) {
        if (visited.find(current_id) != visited.end()) {
            return;
        }
        visited.insert(current_id);
        Node& current_node = g->nodes.find(current_id)->second;
        for (const Edge& neighbor_edge : current_node.neighbors) {
            double neighbor_id = neighbor_edge.to;
            Node& neighbor_node = g->nodes.find(neighbor_id)->second;
            // Only traverse to neighbors that are one move ahead in the game
            if (neighbor_node.data->representation.length() == current_node.data->representation.length() + 1) {
                const_cast<Edge&>(neighbor_edge).opacity = 1;
                dfs(neighbor_id);
            }
        }
    };
    dfs(start_node_id);
}

void hardest_openings(CompositeScene& cs) {
    shared_ptr<C4GraphScene> weakc4 = dynamic_pointer_cast<C4GraphScene>(cs.get_subscene_pointer("weakc4"));
    shared_ptr<Graph> g = weakc4->graph;

    cs.stage_macroblock(FileBlock("This graph serves as a map of Connect 4 openings."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Long branches represent openings which don't quickly simplify for player 1."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("They correspond quite well to openings which Connect 4 players actually study in practice."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And this is great- it provides a framework to measure the difficulty of openings."), 1);
    cs.render_microblock();

    C4Board b(FULL, "4");
    string variation = "15657";
    cs.stage_macroblock(CompositeBlock(FileBlock("For example, if player 2 plays like this, with player 1 following the solution,"), SilenceBlock(4)), 1 + variation.length());
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("4", .5, .5);
    cs.add_scene(c4s, "c4s", -.25, .25);
    cs.slide_subscene(MICRO, "c4s", .5, 0);
    cs.render_microblock();

    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        double node_id = b.get_hash();
        if(!g->node_exists(node_id))
            node_id = b.get_mirror_board().get_hash();
        if(!g->node_exists(node_id))
            throw runtime_error("Node not found in graph!");

        flood_fill_edges_to_highlight(g, node_id);

        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("We run into a steady state right away, meaning Yellow didn't play a very challenging opening."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What about the hardest opening for player 1?"), 1);
    c4s->undo(variation.length());
    // Reset graph edge opacities
    for (auto& pair : g->nodes) {
        Node& node = pair.second;
        for (const Edge& edge : node.neighbors) {
            const_cast<Edge&>(edge).opacity = 1;
        }
    }
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The first move which preserves the longest downstream subgraph is..."), 1);
    cs.render_microblock();

    b = C4Board(FULL, "4");

    c4s->play("3");
    b.play_piece(3);
    double node_id = b.get_hash();
    if(!g->node_exists(node_id))
        node_id = b.get_mirror_board().get_hash();
    if(!g->node_exists(node_id))
        throw runtime_error("Node not found in graph!");

    flood_fill_edges_to_highlight(g, node_id);
    cs.stage_macroblock(FileBlock("One-off from center!"), 1);
    cs.render_microblock();

    variation = "667555";
    cs.stage_macroblock(FileBlock("Red, following the solution, is left with 2727 downstream nodes."), 1 + variation.length());
    cs.render_microblock();

    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        double node_id = b.get_hash();
        if(!g->node_exists(node_id))
            node_id = b.get_mirror_board().get_hash();
        if(!g->node_exists(node_id))
            throw runtime_error("Node not found in graph!");

        flood_fill_edges_to_highlight(g, node_id);

        cs.render_microblock();
    }
    cs.stage_macroblock(FileBlock("Continuing this line of reasoning, the beginning of the 'worst opening' goes like this."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now, if you're a connect 4 player, this probably isn't surprising. This is a very popular, very challenging opening."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    c4s->undo(variation.length() + 1);
    // Reset graph edge opacities
    for (auto& pair : g->nodes) {
        Node& node = pair.second;
        for (const Edge& edge : node.neighbors) {
            const_cast<Edge&>(edge).opacity = 1;
        }
    }
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A close second place goes to the center-column opening. These first 5 moves are entirely forced."), 4);
    variation = "4444";
    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        double node_id = b.get_hash();
        if(!g->node_exists(node_id))
            node_id = b.get_mirror_board().get_hash();
        if(!g->node_exists(node_id))
            throw runtime_error("Node not found in graph!");

        flood_fill_edges_to_highlight(g, node_id);

        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("From here, a large amount of nodes are dedicated to variations of the candlesticks opening, which involves making towers like so, which is also mostly forced."), 8);
    variation = "66222266";
    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        double node_id = b.get_hash();
        if(!g->node_exists(node_id))
            node_id = b.get_mirror_board().get_hash();
        if(!g->node_exists(node_id))
            throw runtime_error("Node not found in graph!");

        flood_fill_edges_to_highlight(g, node_id);

        cs.render_microblock();
    }
}

void anki(CompositeScene& cs) {
    shared_ptr<Mp4Scene> anki_clip = make_shared<Mp4Scene>(vector<string>{"anki_connect4"});
    cs.add_scene(anki_clip, "anki_clip");
    cs.stage_macroblock(FileBlock("Having created a connect 4 strategy which I claim is human-memorizable,"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("I'm putting my money where my mouth is-"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("I made a deck using the spaced-repetition flash card app Anki, which I will attempt to memorize!"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("If you want to learn too, you can download the deck in the description!"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes();
    shared_ptr<Mp4Scene> site_clip = make_shared<Mp4Scene>(vector<string>{"connect4_website"});
    cs.add_scene(site_clip, "site_clip");
    cs.stage_macroblock(FileBlock("I've also included a website where you can play against the weak solution,"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("and explore the tree as you traverse it."), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("Good luck winning... you'll need it!"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("The site includes a lot of technical details which I glazed over, so be sure to check it out!"), 1);
    cs.render_microblock();
}

void render_video() {
    VIDEO_BACKGROUND_COLOR = 0xff000022;
    CompositeScene cs;
    /*
    intro(cs);
    build_graph(cs);
    explanation(cs);
    trees(cs);
    */
    FOR_REAL = false;
    patterned(cs);
    trimmed_solution(cs);
    FOR_REAL = true;
    hardest_openings(cs);
}
