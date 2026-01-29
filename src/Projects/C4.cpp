#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/PauseScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Media/Mp4Scene.cpp"
#include "../Scenes/Media/WhitePaperScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Math/BarChartScene.cpp"
#include "../Scenes/Connect4/C4Scene.cpp"
#include "../Scenes/Media/SvgScene.cpp"
#include "../Scenes/Connect4/C4GraphScene.cpp"
#include "../Scenes/Physics/BouncingBallScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"
// TODO answer Tromp's questions

void intro(CompositeScene& cs) {
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    shared_ptr<PngScene> sky = make_shared<PngScene>("Sky");
    shared_ptr<PngScene> god1 = make_shared<PngScene>("God1", .3, .6);
    shared_ptr<PngScene> god2 = make_shared<PngScene>("God2", .4, .8);

    stage_macroblock(FileBlock("Suppose two omniscient gods play a game of Connect 4."), 2);
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

    stage_macroblock(FileBlock("Players take turns dropping discs in the columns,"), 1);
    c4s->play("433335526245");
    cs.render_microblock();

    stage_macroblock(FileBlock("and the first to make a line of 4 wins."), 3);
    c4s->play("4");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "1");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "0");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(.5), FileBlock("What would happen?")), 1);
    c4s->undo(100);
    cs.render_microblock();

    stage_macroblock(FileBlock("God 1, playing red, plays the first piece in the center column."), 2);
    StateSet god1_opacity = cs.manager.transition(MICRO, "god1.opacity", "1");
    cs.render_microblock();
    c4s->play("4");
    cs.render_microblock();

    stage_macroblock(FileBlock("God 2, playing yellow, promptly resigns."), 3);
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
    stage_macroblock(FileBlock("After analyzing every possible variation of every opening, God 2 realized there was no way of stopping God 1 from making a red line of 4."), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4gs");

    stage_macroblock(FileBlock("This was first discovered by computer scientists in 1988."), 2);
    shared_ptr<PngScene> JamesDowAllen = make_shared<PngScene>("JDA", .4, .8);
    cs.add_scene(JamesDowAllen, "JDA", .25, 1.45);
    cs.slide_subscene(MICRO, "JDA", 0, -1);
    shared_ptr<LatexScene> ls_jda = make_shared<LatexScene>("\\text{James Dow Allen}", 1, .45, .3);
    cs.add_scene_fade_in(MICRO, ls_jda, "ls_jda", .25, .9);
    cs.render_microblock();
    shared_ptr<PngScene> VictorAllis = make_shared<PngScene>("VictorAllis", .4, .8);
    cs.add_scene(VictorAllis, "VA", .75, -.55);
    cs.slide_subscene(MICRO, "VA", 0, 1);
    shared_ptr<LatexScene> ls_va = make_shared<LatexScene>("\\text{Victor Allis}", 1, .45, .3);
    // TODO email and ask permission to use photo
    cs.add_scene_fade_in(MICRO, ls_va, "ls_va", .75, .9);
    cs.render_microblock();

    cs.fade_all_subscenes_except(MICRO, "c4gs", 0);
    stage_macroblock(FileBlock("They used a strategy similar to God 2:"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4gs");

    stage_macroblock(FileBlock("using computer programs to search all possible variations,"), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("showing that player 1 will win, if they play perfectly."), 1);
    cs.render_microblock();

    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", vector<int>{1, 7, 10, 82});
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.add_scene(wps, "wps");
    stage_macroblock(FileBlock("Now, don't get me wrong, this was wonderful work by the computer scientists of the day,"), 1);
    cs.render_microblock();
    cs.remove_subscene("c4gs");

    stage_macroblock(FileBlock("But it kind of leaves you wanting."), 1);
    wps->manager.transition(MICRO, "completion", "0", true);
    cs.render_microblock();
    cs.remove_subscene("wps");

    c4s = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MICRO, c4s, "c4");
    stage_macroblock(FileBlock("Player 1 wins, but _why_?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    god2 = make_shared<PngScene>("God2", .4, .8);
    cs.add_scene(god2, "god2_notice", .75, -.5);
    cs.manager.transition(MICRO, floating_gods);
    //TODO need some animations here
    stage_macroblock(FileBlock("A computer, or a god, can iterate over billions of positions to check this result,"), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("But what's left for us humans?"), 1);
    cs.fade_subscene(MICRO, "god2", 0);
    shared_ptr<PngScene> human1 = make_shared<PngScene>("Thinker1", .3, .6);
    shared_ptr<PngScene> human2 = make_shared<PngScene>("Thinker2", .3, .6);
    cs.add_scene(human1, "human1", .15, 1.7);
    cs.add_scene(human2, "human2", .85, 1.7);
    cs.slide_subscene(MICRO, "human1", 0, -1);
    cs.slide_subscene(MICRO, "human2", 0, -1);
    cs.render_microblock();
    cs.remove_subscene("god2");

    cs.fade_subscene(MICRO, "human2", .2);
    stage_macroblock(FileBlock("After player 1 plays in the center, what player 2 openings make it as hard as possible for player 1?"), 7);
    cs.render_microblock();
    cs.render_microblock();
    c4s->play("4");
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "human2", 1);
    cs.fade_subscene(MICRO, "human1", .2);
    shared_ptr<PngScene> human2_trick = make_shared<PngScene>("Thinker2_trick", .3, .6);
    cs.add_scene_fade_in(MICRO, human2_trick, "human2_trick", .85, .7);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("To confuse red, should player 2 respond in the center, or one over? What about all the way on the side?"), 8);
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
    cs.render_microblock();

    // TODO fading is wrong here
    stage_macroblock(FileBlock("Is there some change of perspective that shows how us mortals could hang on to the player one advantage, and defeat god 2?"), 4);
    cs.slide_subscene(MICRO, "human2", 0, 1);
    cs.render_microblock();
    cs.remove_subscene("human2");
    cs.fade_subscene(MICRO, "human1", 1);
    cs.render_microblock();
    cs.render_microblock();
    god2 = make_shared<PngScene>("God2", .4, .8);
    cs.add_scene_fade_in(MICRO, god2, "god2", .75, -.5);
    cs.manager.transition(MICRO, floating_gods);
    cs.render_microblock();

    stage_macroblock(FileBlock("These questions went unanswered... until now."), 1);
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

    stage_macroblock(FileBlock("Because today, I'm going to show you connect 4 through the eyes of God 1."), 2);
    cs.render_microblock();
    cs.render_microblock();

    gs->manager.transition(MICRO, {{"q1", "1"}, {"qj", "0"}, {"qi", "{t} 3 / sin"}, {"qk", "0"}});
    stage_macroblock(FileBlock("Because I found a profoundly simpler way to play perfect connect 4. All it takes is this graph."), 1);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3), 1);
    gs->manager.transition(MICRO, "d", "-1");
    gs->manager.transition(MICRO, "physics_multiplier", "0");
    cs.fade_all_subscenes_except(MICRO, "gs", 0);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

void explanation(CompositeScene& cs) {
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MACRO, c4s, "c4s");
    stage_macroblock(FileBlock("Suppose you're going to play as red, and you want a strategy to play perfectly."), 3);
    cs.render_microblock();
    string starting_variation = "4443674433";
    c4s->play(starting_variation);
    cs.render_microblock();
    cs.render_microblock();

    c4s->set_fast_mode(true);
    int vars_to_read = 5;
    int vars_depth = 20;
    stage_macroblock(FileBlock("What options are there, besides running brute force search in your head?"), vars_to_read * 2);
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

    stage_macroblock(FileBlock("Instead you could memorize every branch upfront."), 1);
    cs.fade_all_subscenes(MICRO, 0);
    unordered_map<string, string> branches = {
        {"44436445527455", "Play in third column"},
        {"42656566544", "Play in fourth column"},
        {"42656466442422361717", "Play in first column"},
        {"4365567256556445", "Play in third column"},
        {"4523321225553336", "Play in third column"},
        {"45212244174556", "Play in seventh column"},
        {"46251324442252", "Play in fifth column"},
    };
    cs.manager.begin_timer("shift");
    int which_branch = 0;
    for(const auto& pair : branches) {
        shared_ptr<C4Scene> branch_scene = make_shared<C4Scene>("", .5, .5);
        branch_scene->set_fast_mode(true);
        branch_scene->play(pair.first);
        shared_ptr<LatexScene> branch_label = make_shared<LatexScene>("\\text{" + pair.second + "}", 1, .5, .5);
        cs.add_scene_fade_in(MICRO, branch_scene, "branch_scene"+to_string(which_branch));
        cs.add_scene_fade_in(MICRO, branch_label, "branch_label"+to_string(which_branch));
        cs.manager.set({
            {"branch_scene" + to_string(which_branch) + ".x", ".25"},
            {"branch_scene" + to_string(which_branch) + ".y", ".25 " + to_string(which_branch) + " .5 * + <shift> +"},
            {"branch_label" + to_string(which_branch) + ".x", ".75"},
            {"branch_label" + to_string(which_branch) + ".y", ".25 " + to_string(which_branch) + " .5 * + <shift> +"},
        });
        which_branch++;
    }
    cs.render_microblock();

    cs.fade_all_subscenes(MICRO, .3);
    cs.manager.begin_timer("shift_2");
    which_branch = 0;
    for(const auto& pair : branches) {
        shared_ptr<C4Scene> branch_scene = make_shared<C4Scene>("", .5, .5);
        branch_scene->set_fast_mode(true);
        branch_scene->play(pair.first);
        shared_ptr<LatexScene> branch_label = make_shared<LatexScene>("\\text{" + pair.second + "}", 1, .5, .5);
        cs.add_scene_fade_in(MICRO, branch_scene, "branch_scene_"+to_string(which_branch));
        cs.add_scene_fade_in(MICRO, branch_label, "branch_label_"+to_string(which_branch));
        cs.manager.set({
            {"branch_scene" + to_string(which_branch) + ".x", ".25 " + to_string(which_branch) + " .5 * + <shift_2> +"},
            {"branch_scene" + to_string(which_branch) + ".y", ".25"},
            {"branch_label" + to_string(which_branch) + ".x", ".25 " + to_string(which_branch) + " .5 * + <shift_2> +"},
            {"branch_label" + to_string(which_branch) + ".y", ".75"},
        });
        which_branch++;
    }
    stage_macroblock(FileBlock("That way you can just recall the right response during the game."), 1);
    cs.render_microblock();

    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<WhitePaperScene> bock = make_shared<WhitePaperScene>("bock_paper", "Böck (2025)" vector<int>{1, 2, 3, 4});
    cs.add_scene(bock, "bock");
    bock->manager.transition(MICRO, "completion", "1", false);
    bock->manager.set("which_page", "1");
    stage_macroblock(FileBlock("Well, you'll need to memorize 90 gigabytes of connect 4 data, as this poor soul found out."), 4);
    cs.render_microblock();
    bock->manager.transition(MICRO, "page_focus", "1");
    cs.render_microblock();
    cs.remove_all_subscenes_except("bock");
    bock->manager.transition(MICRO, {
        {"crop_top", ".373"},
        {"crop_bottom", ".612"},
        {"crop_right", ".258"},
        {"crop_left", ".371"},
    });
    cs.render_microblock();
    cs.render_microblock();

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(g, false, "", FULL);

    cs.fade_all_subscenes(MICRO, 0);
    cs.add_scene(c4gs, "c4gs");
    stage_macroblock(FileBlock("These two strategies- brute force search and upfront memorization- involve the same tree of positions."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("The difference is that the former makes you think a lot, and the latter makes you remember a lot."), 2);
    cs.fade_subscene(MICRO, "c4gs", 0);
    shared_ptr<PngScene> cpu = make_shared<PngScene>("cpu", .4, .4);
    cs.add_scene_fade_in(MICRO, cpu, "cpu", .25, .5);
    cs.render_microblock();
    cs.remove_subscene("c4gs");
    shared_ptr<PngScene> hdd = make_shared<PngScene>("hdd", .4, .4);
    cs.add_scene_fade_in(MICRO, hdd, "hdd", .75, .5);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
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

    stage_macroblock(FileBlock("Plotting the amount of positions up to n moves, you get a curve like this."), 2);
    rfs->manager.transition(MICRO, "function0_right", "20");
    cs.render_microblock();

    cs.fade_subscene(MICRO, "rfs", 0.3);
    shared_ptr<C4Scene> quick_display = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MICRO, quick_display, "quick_display");
    quick_display->set_fast_mode(true);
    cs.render_microblock();

    stage_macroblock(FileBlock("There's 7 options for the first move,"), 8);
    for(int i = 1; i <= 7; i++) {
        quick_display->undo(1);
        quick_display->play(to_string(i));
        cs.render_microblock();
    }
    quick_display->undo(1);
    cs.render_microblock();

    stage_macroblock(FileBlock("after 5 moves there's 4000 possibilities."), 1);
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
    quick_display->flush_queue();

    stage_macroblock(FileBlock("and after 10, there's over a million."), 1);
    quick_display->undo(5);
    for(int i = 0; i < 1000; i++) {
        string random_sequence = "";
        for(int j = 0; j < 10; j++){
            char ch = (char)(rand() % 7 + '1');
            random_sequence += ch;
        }
        quick_display->play(random_sequence);
        quick_display->undo(10);
    }
    cs.render_microblock();

    stage_macroblock(FileBlock("Classic exponential growth!"), 1);
    cs.slide_subscene(MICRO, "quick_display", 0, 1);
    cs.fade_subscene(MICRO, "rfs", 1);
    cs.render_microblock();
    cs.remove_subscene("quick_display");

    stage_macroblock(FileBlock("Now, after n moves have already transpired, if we plot the amount of variations that need to be searched to brute-force solve the rest of the game..."), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function1_right", "20");

    // TODO barren of animation

    stage_macroblock(FileBlock("we see an opposite curve. It's easier to work out a game that's almost over than one which has just started."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("This suggests a hybrid approach-"), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function0_right", "<center_x>");
    stage_macroblock(FileBlock("Memorizing all positions only up to the midgame,"), 1);
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function1_left", "<center_x>");
    stage_macroblock(FileBlock("and then once we are halfway through, we switch over to reading ahead."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("In other words, we can balance compute and memory to avoid these exponential explosions."), 1);
    cs.render_microblock();

    cs.fade_all_subscenes(MICRO, 0);
    shared_ptr<LatexScene> ls_memo = make_shared<LatexScene>("\\text{Common opening worth memorizing}", 1, .8, .3);
    shared_ptr<LatexScene> ls_comp = make_shared<LatexScene>("\\text{Rare variation, switch to reading ahead}", 1, .8, .3);
    cs.add_scene_fade_in(MICRO, ls_memo, "ls_memo", .5, .1);
    shared_ptr<Mp4Scene> chess_clip = make_shared<Mp4Scene>(vector<string>{"chess"}, 3, .6, .6);
    cs.add_scene_fade_in(MICRO, chess_clip, "chess_clip");
    StateSet frame = chess_clip->manager.set("current_frame", "0");
    stage_macroblock(FileBlock("This is why chess players memorize openings, and start reading after the position has developed beyond their memory."), 5);
    cs.render_microblock();
    cs.manager.begin_timer("MP4_Frame");
    cs.manager.set(frame);
    chess_clip->manager.set("current_frame", "[current_frame]");
    cs.add_scene(ls_comp, "ls_comp", .5, .1);
    cs.manager.set({
        {"ls_memo.x", ".5 <current_frame> 230 - 30 / max"},
        {"ls_comp.x", ".5 <current_frame> 250 - 30 / min"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", "Allis (1988)" vector<int>{1, 7, 10, 82});
    cs.add_scene(wps, "wps");
    stage_macroblock(FileBlock("Victor Allis published such a solution in 1988: a table of 500,000 perfect openings, and a computer algorithm to solve endgames in a few seconds."), 8);
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.set("which_page", "82");
    wps->manager.transition(MICRO, "page_focus", "1");
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".253"}, {"crop_bottom", ".707"}});
    shared_ptr<WhitePaperScene> wps_highlight = make_shared<WhitePaperScene>("allis_paper", "Allis (1988)" vector<int>{1, 7, 10, 83});
    wps_highlight->manager.set({{"completion", "1"}, {"page_focus", "1"}, {"which_page", "83"}});
    wps_highlight->manager.transition(MICRO, {{"crop_top", ".253"}, {"crop_bottom", ".707"}});
    cs.add_scene_fade_in(MICRO, wps_highlight, "wps_highlight");
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".41"}, {"crop_bottom", ".572"}, {"crop_left", ".1"}, {"crop_right", ".4"}});
    wps_highlight->manager.transition(MICRO, {{"crop_top", ".41"}, {"crop_bottom", ".572"}, {"crop_left", ".1"}, {"crop_right", ".4"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("wps_highlight");

    stage_macroblock(FileBlock("Now I don't know about you, but I can't memorize half a million positions."), 2);
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

void trees(CompositeScene& cs) {
    shared_ptr<Graph> g1 = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(g1, false, "", FULL);
    c4gs->manager.set("points_radius_multiplier", "0.5");
    c4gs->manager.set("desired_nodes", "5678 1.5 <time_since_graph_init> ^ 200 * 199 - min");
    cs.add_scene(c4gs, "c4gs");
    stage_macroblock(FileBlock("Before I reveal my trick which turns half-a-million into half-a-ten-thousand, let me explain the various types of trees that I've been showing in the background."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("If we add every possible connect 4 position to a tree, we get this gigantic thing."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("This full game tree up to move 5, taken as a guide on where to play, is called a 'strong solution'."), 2);
    cs.render_microblock();
    shared_ptr<LatexScene> ls_strong = make_shared<LatexScene>("\\text{Strong Solution {\\tiny (Depth 5)}}", 1, .5, .2);
    cs.add_scene_fade_in(MICRO, ls_strong, "ls_strong", .5, .1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ls_strong", 0);
    stage_macroblock(FileBlock("However, this tree includes nodes where player 1, red, has already blundered."), 1);
    cs.render_microblock();
    cs.remove_subscene("ls_strong");
    
    stage_macroblock(FileBlock("We don't want to memorize those, since we'd never play them."), 1);
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
    stage_macroblock(FileBlock("So let's grab our hedge-trimmers and go to work..."), to_delete.size());
    for(const double& id : to_delete) {
        g1->remove_node(id);
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("We get this smaller tree."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("We can shrink it further though."), 1);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("45");
    cs.add_scene(c4s, "c4s", 1.5, .5);
    cs.slide_subscene(MICRO, "c4s", -.75, 0);
    cs.slide_subscene(MICRO, "c4gs", -.25, 0);
    stage_macroblock(FileBlock("Consider this position."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("It's red's move, and there's two winning options."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Red could either go here,"), 2);
    c4s->play("2");
    cs.render_microblock();
    c4s->undo(1);
    cs.render_microblock();

    stage_macroblock(FileBlock("or here."), 1);
    c4s->play("1");
    cs.render_microblock();

    stage_macroblock(FileBlock("Everything else would be a blunder."), 1);
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
            it->second.draw_point = true;
        }
    }
    stage_macroblock(FileBlock("Those two nodes are in our tree."), 1);
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

    stage_macroblock(FileBlock("But why memorize duplicates, instead of just memorizing the one we like better?"), 2);
    cs.render_microblock();
    for(const double& id : to_delete) {
        g1->remove_node(id);
    }
    c4gs->manager.transition(MICRO, "points_opacity", "0");
    cs.render_microblock();

    // Delete all nodes not in c4gs_weak
    to_delete.clear();
    it = g1->nodes.begin();
    for (; it != g1->nodes.end(); ++it) {
        it->second.size = 1;
        if(g3->nodes.find(it->first) == g3->nodes.end()) {
            to_delete.push_back(it->first);
            it->second.draw_point = false;
        }
    }

    c4gs->manager.set("points_opacity", "1");

    stage_macroblock(FileBlock("Let's delete each duplicate option."), to_delete.size() + 1);
    cs.render_microblock();
    for(const double& id : to_delete) {
        g1->remove_node(id);
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("This tree is sufficient- we can still play perfectly up to move 5 by memorizing it. We've just cut out all the stupid variations that we don't need."), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls_weak = make_shared<LatexScene>("\\text{Weak Solution {\\tiny (Depth 5)}}", 1, .5, .2);
    cs.add_scene_fade_in(MICRO, ls_weak, "ls_weak", .5, .1);
    stage_macroblock(FileBlock("It's called a 'weak solution'."), 1);
    cs.render_microblock();

    //TODO answer: who cares?

    cs.fade_all_subscenes(MICRO, 0);
    stage_macroblock(SilenceBlock(1), 1);
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

double find_node_id_from_board(shared_ptr<Graph> g, C4Board& b) {
    double start_node_id = b.get_hash();
    if(!g->node_exists(start_node_id))
        start_node_id = b.get_mirror_board().get_hash();
    if(!g->node_exists(start_node_id))
        throw runtime_error("Node not found in graph! " + b.representation);
    return start_node_id;
}

void flood_fill_edges_to_highlight(shared_ptr<Graph> g, C4Board& b, shared_ptr<C4GraphScene> gs) {
    double start_node_id = find_node_id_from_board(g, b);

    gs->next_hash = start_node_id;

    // First set all edges to low opacity
    for (auto& pair : g->nodes) {
        Node& node = pair.second;
        for (const Edge& edge : node.neighbors) {
            const_cast<Edge&>(edge).opacity = 0.1;
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

void reset_graph_edge_opacities(shared_ptr<Graph> g) {
    for (auto& pair : g->nodes) {
        Node& node = pair.second;
        for (const Edge& edge : node.neighbors) {
            const_cast<Edge&>(edge).opacity = 1;
        }
    }
}

void patterned(CompositeScene& cs) {
    string variation = "43636335555665773563";
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    cs.add_scene_fade_in(MICRO, c4s, "c4s");
    stage_macroblock(CompositeBlock(CompositeBlock(FileBlock("Check out this game."), SilenceBlock(2)), FileBlock("It's almost over, with Red to move.")), 5);
    cs.render_microblock();
    c4s->play(variation);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    c4s->manager.transition(MICRO, {{"w", ".3"}, {"h", ".3"}});
    cs.slide_subscene(MICRO, "c4s", 0, -.35);

    stage_macroblock(FileBlock("Now what do you think the weak solution looks like for this endgame?"), 1);
    cs.render_microblock();

    shared_ptr<Graph> g_weak_1 = make_shared<Graph>();
    shared_ptr<Graph> g_weak_2 = make_shared<Graph>();
    shared_ptr<C4GraphScene> c4gs_weak_1 = make_shared<C4GraphScene>(g_weak_1, false, variation, RIGHTMOST_WEAK);
    shared_ptr<C4GraphScene> c4gs_weak_2 = make_shared<C4GraphScene>(g_weak_2, false, variation, SIMPLE_WEAK);
    c4gs_weak_1->manager.set("physics_multiplier", "50");
    c4gs_weak_2->manager.set("physics_multiplier", "50");
    c4gs_weak_1->manager.set("growth_rate", "150");
    c4gs_weak_2->manager.set("growth_rate", "8");
    stage_macroblock(FileBlock("I'll give you 2 options."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Do you think it looks like this messy tree?"), 1);
    cs.add_scene(c4gs_weak_1, "c4gs_weak_1", .25, .5);
    cs.render_microblock();
    stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Or do you think that it looks like this nice pillow shape?"), 1);
    cs.add_scene(c4gs_weak_2, "c4gs_weak_2", .75, .5);
    cs.render_microblock();

    stage_macroblock(FileBlock("This isn't just some trivial question about appearance- if there is a particularly simple weak solution, that means we can compress the information contained within it."), 1);
    cs.render_microblock();

    shared_ptr<PauseScene> ps = make_shared<PauseScene>();
    cs.add_scene(ps, "ps");
    stage_macroblock(CompositeBlock(FileBlock("So, make your guess! Which one do you think it is?"), SilenceBlock(1)), 1);
    cs.render_microblock();

    cs.remove_subscene("ps");
    stage_macroblock(FileBlock("The answer is... both of them!"), 1);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0.2);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s_compare = make_shared<C4Scene>("");
    stage_macroblock(FileBlock("Remember how I chose between two winning options earlier?"), 5);
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

    stage_macroblock(FileBlock("Up to the choices we make, we can get much different trees!"), 1);
    cs.fade_subscene(MICRO, "c4s_compare", 0);
    cs.fade_subscene(MICRO, "c4gs_weak_2", 1);
    cs.fade_subscene(MICRO, "c4gs_weak_1", 1);
    cs.fade_subscene(MICRO, "c4s", 1);
    cs.render_microblock();
    cs.remove_subscene("c4s_compare");

    c4gs_weak_1->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.slide_subscene(MICRO, "c4gs_weak_1", .25, 0);
    cs.fade_subscene(MICRO, "c4gs_weak_2", 0);
    stage_macroblock(FileBlock("I made this one by always choosing the rightmost winning column."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "c4gs_weak_1", 0);
    cs.fade_subscene(MICRO, "c4gs_weak_2", 1);
    cs.render_microblock();
    cs.remove_subscene("c4gs_weak_1");

    c4gs_weak_2->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.slide_subscene(MICRO, "c4gs_weak_2", -.25, 0);
    stage_macroblock(FileBlock("So how on earth did I make this one?"), 1);
    cs.render_microblock();

    cs.fade_all_subscenes_except(MICRO, "c4s", 0);
    c4s->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.slide_subscene(MICRO, "c4s", 0, .35);
    stage_macroblock(FileBlock("How do we identify these patterny weak solutions?"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4s");

    stage_macroblock(FileBlock("I use what I call a 'steady state diagram'."), 2);
    c4s->undo(100);
    c4s->play("473534");
    cs.render_microblock();
    c4s->set_annotations_from_steadystate(MICRO);
    cs.render_microblock();

    cs.slide_subscene(MICRO, "c4s", .2, 0);
    stage_macroblock(FileBlock("The diagram is like a cheat-sheet, telling Red how to make all the right moves from here until the end of the game."), 1);
    cs.render_microblock();

    shared_ptr<SvgScene> rules = make_shared<SvgScene>("steady_state_rules_0", 1, .5, 1);
    cs.add_scene(rules, "rules", .25, .5);
    stage_macroblock(FileBlock("To read it, there's a series of 8 priorities."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("It's red's turn. Let's follow the diagram to win the game."), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_1");
    stage_macroblock(FileBlock("Rule 1: Is there a winning move available?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    stage_macroblock(FileBlock("Nope. Move on to rule 2."), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_2");
    stage_macroblock(FileBlock("Is the opponent about to win, such that we need to block it?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    stage_macroblock(FileBlock("Nope. Continue!"), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_3");
    stage_macroblock(FileBlock("Can we play on an exclamation mark?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    stage_macroblock(FileBlock("Yes! That's the move we'll take."), 2);
    c4s->play("3");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Our opponent blocks the threat. What should we do now?"), 1);
    c4s->play("3");
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_6");
    stage_macroblock(FileBlock("The first rule that applies now is rule 6, which says to play on a plus sign."), 4);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    c4s->play("2");
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_0");
    stage_macroblock(FileBlock("Our opponent plays here."), 2);
    cs.render_microblock();
    c4s->play("1");
    cs.render_microblock();

    stage_macroblock(FileBlock("Try and figure out what Red should do next, according to the rules."), 1);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    rules->begin_svg_transition(MICRO, "steady_state_rules_4");
    stage_macroblock(FileBlock("There's only one at sign available now, so Red plays on it."), 2);
    cs.render_microblock();
    c4s->play("4");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("As Red, if you follow this cheat sheet, regardless of what Yellow does, you'll eventually win the game."), SilenceBlock(2)), 14);
    c4s->play("41");
    cs.render_microblock();
    c4s->play("33");
    cs.render_microblock();
    c4s->play("57");
    cs.render_microblock();
    c4s->play("44");
    cs.render_microblock();
    c4s->play("77");
    cs.render_microblock();
    c4s->play("66");
    cs.render_microblock();
    c4s->play("66");
    cs.render_microblock();
    c4s->play("11");
    cs.render_microblock();
    c4s->play("66");
    cs.render_microblock();
    c4s->play("11");
    cs.render_microblock();
    c4s->play("77");
    cs.render_microblock();
    c4s->play("55");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "1");
    cs.render_microblock();
    c4s->manager.transition(MICRO, "highlight", "0");
    cs.render_microblock();

    cs.fade_all_subscenes_except(MICRO, "c4s", 0);
    stage_macroblock(SilenceBlock(2), 1);
    c4s->clear_annotations(MICRO);
    c4s->undo(100);
    cs.render_microblock();
    cs.remove_all_subscenes_except("c4s");

    stage_macroblock(FileBlock("These diagrams are effectively a form of compression."), 1);
    c4s->play(variation);
    cs.render_microblock();
    c4s->set_annotations_from_steadystate(MICRO);

    stage_macroblock(FileBlock("By identifying such a simple program to solve a connect 4 position, we can memorize that program instead of some convoluted tree traversed by brute force search."), 1);
    cs.add_scene_fade_in(MICRO, c4gs_weak_1, "c4gs_weak_1", .25, .5);
    cs.render_microblock();

    cs.slide_subscene(MICRO, "c4s", -.2, 0);
    cs.fade_subscene(MICRO, "c4gs_weak_1", 0);
    stage_macroblock(FileBlock("Now, one obvious question is: what is the cheat sheet for the starting position?"), 1);
    c4s->undo(variation.length());
    c4s->clear_annotations(MICRO);
    cs.render_microblock();
    cs.remove_subscene("c4gs_weak_1");

    stage_macroblock(FileBlock("Unfortunately we're not that lucky."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("These diagrams are only good for describing midgames when the path to victory is already somewhat well-defined."), 2);
    c4s->play("44444221");
    cs.render_microblock();
    c4s->set_annotations_from_steadystate(MICRO);
    cs.render_microblock();
}

void trimmed_solution(CompositeScene& cs) {
    stage_macroblock(FileBlock("So we're back to building trees."), 2);
    cs.fade_subscene(MICRO, "c4s", 0);
    cs.render_microblock();
    cs.remove_all_subscenes();

    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<C4GraphScene> weakc4 = make_shared<C4GraphScene>(g, false, "", TRIM_STEADY_STATES);
    weakc4->manager.set({{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}, {"mirror_force",".005"}});
    if (!rendering_on()) {
        g->expand(-1);
        g->make_bidirectional();
    }
    cs.add_scene(weakc4, "weakc4");
    weakc4->manager.set("physics_multiplier", "60");
    cs.render_microblock();

    stage_macroblock(FileBlock("But this time, we can terminate search on positions which have a steady state solution."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("And if we furthermore try to find the smallest such graph- that is, the one that most quickly directs you to those easy cheat-sheet positions, you get something like this."), 1);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s_steady = make_shared<C4Scene>("", .5, .5);
    string variation = "444145552244122233";
    stage_macroblock(FileBlock("So, to be clear, this is once again a weak solution. But, at every leaf node, instead of the game being over,"), variation.length() + 1);
    cs.add_scene_fade_in(MICRO, c4s_steady, "c4s_steady", .2, .2);
    weakc4->manager.transition(MICRO, {{"q1", "0"}, {"qj", "0"}, {"qi", "{t} 3 / sin 10 /"}, {"qk", "0"}, {"d", ".5"}});
    cs.render_microblock();

    C4Board b(FULL, "");
    for(char c : variation) {
        string move(1, c);
        c4s_steady->play(move);
        b.play_piece(move[0] - '0');

        flood_fill_edges_to_highlight(g, b, weakc4);

        cs.render_microblock();
    }

    stage_macroblock(FileBlock("we instead have a steady state diagram showing how to continue perfect play after that point, implicitly defining the remaining compressed game tree."), 2);
    c4s_steady->set_annotations_from_steadystate(MICRO);
    cs.render_microblock();
    double node_id = find_node_id_from_board(g, b);
    g->remove_node(node_id); // remove so we can regrow it
    weakc4->manager.begin_timer("subtree_grow");
    weakc4->manager.set("desired_nodes", to_string(g->size()) + " <subtree_grow> 100 * +");
    shared_ptr<SteadyState> ss = find_steady_state(variation, nullptr, true);
    g->add_to_stack(new C4Board(C4BranchMode::SIMPLE_WEAK, variation, ss));
    cs.render_microblock();
    if(!rendering_on()) {
        g->expand(-1);
        g->make_bidirectional();
    }

    b = C4Board(FULL, "");
    weakc4->next_hash = b.get_hash();
    cs.fade_subscene(MICRO, "c4s_steady", 0);
    g->remove_node(node_id); // remove so we can regrow it
    g->add_to_stack(new C4Board(C4BranchMode::FULL, variation));
    unordered_set<double> to_delete;
    for(auto &pair : g->nodes) {
        if(dynamic_cast<C4Board*>(pair.second.data)->c4_branch_mode == C4BranchMode::SIMPLE_WEAK) {
            to_delete.insert(pair.first);
        }
    }
    weakc4->manager.set("desired_nodes", "0");
    stage_macroblock(SilenceBlock(1), to_delete.size());
    for(double node_id : to_delete) {
        g->remove_node(node_id);
        cs.render_microblock();
    }
    cs.remove_subscene("c4s_steady");

    weakc4->manager.transition(MICRO, {{"q1", "1"}, {"qj", "0"}, {"qi", "{t} 10 / sin 10 /"}, {"qk", "0"}});
    weakc4->manager.set("physics_multiplier", "0");
    stage_macroblock(FileBlock("The first thing you might notice is that it's symmetrical."), 1);
    cs.render_microblock();

    shared_ptr<C4Scene> c4s_left = make_shared<C4Scene>("", .5, 1);
    shared_ptr<C4Scene> c4s_right = make_shared<C4Scene>("", .5, 1);
    cs.add_scene(c4s_left, "c4s_left", -.25, .5);
    cs.add_scene(c4s_right, "c4s_right", 1.25, .5);
    cs.slide_subscene(MICRO, "c4s_left", .5, 0);
    cs.slide_subscene(MICRO, "c4s_right", -.5, 0);
    stage_macroblock(FileBlock("This makes sense- If yellow plays on the left versus on the right, we'll respond in a way that is mirrored horizontally."), 4);
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

    stage_macroblock(FileBlock("There's a few exceptions to this rule. For example, this section only has one asymmetric copy."), 3);
    cs.slide_subscene(MICRO, "c4s_left", -.5, 0);
    cs.slide_subscene(MICRO, "c4s_right", .5, 0);
    cs.render_microblock();
    cs.remove_subscene("c4s_left");
    cs.remove_subscene("c4s_right");

    C4Board asymm_board(FULL, "4444445");
    flood_fill_edges_to_highlight(g, asymm_board, weakc4);
    double hash = asymm_board.get_hash();
    glm::vec4 pos = g->nodes.find(hash)->second.position;
    weakc4->manager.transition(MICRO, {{"x", to_string(pos.x)}, {"y", to_string(pos.y)}, {"z", to_string(pos.z)}, {"d", ".5"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("In this case, both players made a perfectly symmetrical opening, and Red must pick a side to continue on."), 5);
    shared_ptr<C4Scene> c4s_asymm = make_shared<C4Scene>("", .5, .5);
    cs.add_scene_fade_in(MICRO, c4s_asymm, "c4s_asymm", .25, .25);
    cs.render_microblock();
    c4s_asymm->play("444444");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    c4s_asymm->play("5");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("This opening is called the 6-1, because of the central column of 6 and neighboring column of 1."), SilenceBlock(1)), 3);
    cs.slide_subscene(MICRO, "c4s_asymm", .25, .25);
    c4s_asymm->manager.transition(MICRO, {{"w", "1"}, {"h", "1"}});
    cs.fade_subscene(MICRO, "weakc4", .3);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "c4s_asymm", 0);
    cs.fade_subscene(MICRO, "weakc4", 1);
    cs.render_microblock();
    cs.remove_subscene("c4s_asymm");
    reset_graph_edge_opacities(g);

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
    stage_macroblock(FileBlock("Let's fuse together mirror-symmetric nodes!"), 1);
    weakc4->manager.set("physics_multiplier", "60");
    weakc4->manager.transition(MICRO, {{"x", "0"}, {"y", "0"}, {"z", "0"}, {"d", "1"}});
    C4Board empty_board(FULL, "");
    weakc4->next_hash = empty_board.get_hash();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3), sqrt(to_merge.size()) + 5);
    int iter = 0;
    int n = 1;
    weakc4->manager.transition(MACRO, "dimensions", "2");
    for(const auto& p : to_merge) {
        g->collapse_two_nodes(p.first, p.second);
        iter++;
        if(iter != n * n) continue;
        n += 1;
        cs.render_microblock();
    }
    while(remaining_microblocks_in_macroblock) {
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("This connect 4 solution has only 4,550 nodes!"), 1);
    cs.render_microblock();


    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("allis_paper", vector<int>{1, 7, 10, 82});
    cs.add_scene(wps, "wps");
    stage_macroblock(FileBlock("Comparing against Victor Allis's solution from the eighties,"), 2);
    wps->manager.transition(MICRO, "completion", "1", false);
    cs.render_microblock();
    wps->manager.set("which_page", "82");
    wps->manager.transition(MICRO, "page_focus", "1");
    cs.render_microblock();

    weakc4->manager.transition(MICRO, "physics_multiplier", "0");
    stage_macroblock(FileBlock("Allis's opening book was more than a hundred times as large, with half a million nodes."), 4);
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});

    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {{"crop_top", ".253"}, {"crop_bottom", ".707"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    wps->manager.transition(MICRO, {{"crop_top", ".133"}, {"crop_bottom", ".558"}, {"crop_left", ".1"}, {"crop_right", ".1"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("Furthermore, unlike the few seconds of search that Allis used,"), 2);
    wps->manager.transition(MICRO, {{"crop_top", ".41"}, {"crop_bottom", ".572"}, {"crop_left", ".1"}, {"crop_right", ".4"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("we don't need any search whatsoever since the steady state diagrams instantly tell us what to do in the endgames."), 1);
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

shared_ptr<C4GraphScene> hardest_openings(CompositeScene& cs) {
    shared_ptr<C4GraphScene> weakc4 = dynamic_pointer_cast<C4GraphScene>(cs.get_subscene_pointer("weakc4"));
    shared_ptr<Graph> g = weakc4->graph;

    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    // TODO highlight openings by name
    stage_macroblock(FileBlock("This graph serves as a map of Connect 4 openings."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Long branches represent openings which don't quickly simplify for player 1."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("They correspond quite well to openings which Connect 4 players actually study in practice."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("And this is great- it provides a framework to measure the difficulty of openings."), 1);
    cs.render_microblock();

    C4Board b(FULL, "4");
    string variation = "15657";
    stage_macroblock(CompositeBlock(FileBlock("For example, if player 2 plays like this, with player 1 following the solution,"), SilenceBlock(4)), 1 + variation.length());
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("4", .7, .7);
    cs.add_scene(c4s, "c4s", -.3, .2);
    cs.slide_subscene(MICRO, "c4s", .5, 0);
    cs.render_microblock();

    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        flood_fill_edges_to_highlight(g, b, weakc4);

        cs.render_microblock();
    }

    stage_macroblock(FileBlock("We run into a steady state right away, meaning Yellow didn't play a very challenging opening."), 2);
    c4s->set_annotations_from_steadystate(MICRO);
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    c4s->undo(variation.length());
    c4s->clear_annotations(MICRO);
    cs.render_microblock();

    stage_macroblock(FileBlock("What about the hardest opening for player 1?"), 1);
    b = C4Board(FULL, "4");
    reset_graph_edge_opacities(g);
    weakc4->next_hash = b.get_hash();
    cs.render_microblock();

    stage_macroblock(FileBlock("The first move which preserves the longest downstream subgraph is..."), 1);
    cs.render_microblock();

    c4s->play("3");
    b.play_piece(3);
    flood_fill_edges_to_highlight(g, b, weakc4);
    stage_macroblock(FileBlock("One-off from center!"), 1);
    cs.render_microblock();

    variation = "667555";
    stage_macroblock(FileBlock("Red, following the solution, is left with 2727 downstream nodes."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Continuing this line of reasoning, the beginning of the 'worst opening' goes like this."), variation.length());
    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        flood_fill_edges_to_highlight(g, b, weakc4);

        cs.render_microblock();
    }

    stage_macroblock(FileBlock("Now, if you're a connect 4 player, this probably isn't surprising. This is a very popular, very challenging opening."), 1);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    c4s->undo(variation.length() + 1);
    c4s->clear_annotations(MICRO);
    b = C4Board(FULL, "4");
    reset_graph_edge_opacities(g);
    weakc4->next_hash = b.get_hash();
    cs.render_microblock();

    stage_macroblock(FileBlock("A close second place goes to the center-column opening. These first 5 moves are entirely forced."), 4);
    variation = "4444";
    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        flood_fill_edges_to_highlight(g, b, weakc4);

        cs.render_microblock();
    }

    stage_macroblock(FileBlock("From here, a large amount of nodes are dedicated to variations of the candlesticks opening, which involves making towers like so, which is also forced."), 8);
    variation = "66662222";
    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        flood_fill_edges_to_highlight(g, b, weakc4);

        cs.render_microblock();
    }

    stage_macroblock(SilenceBlock(2), 1);
    cs.fade_all_subscenes(MICRO, 0);
    C4Board empty_board(FULL, "");
    weakc4->next_hash = empty_board.get_hash();
    cs.render_microblock();
    cs.remove_all_subscenes();
    return weakc4;
}

void solution_types(CompositeScene& cs) {
    stage_macroblock(FileBlock("Now, we've covered a whole bunch of solutions to connect four."), 1);
    cs.render_microblock();

    shared_ptr<BarChartScene> bcs = make_shared<BarChartScene>("Brute Force Search", vector<string>{"Computation", "Memory"});
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
    cs.add_scene_fade_in(MICRO, rfs, "rfs", 0.5, 0.5, 0.4);

    stage_macroblock(FileBlock("Brute force search relies intensively on reading ahead, but needs no memory."), 3);
    cs.add_scene_fade_in(MICRO, bcs, "bcs");
    cs.render_microblock();
    bcs->manager.transition(MICRO, {{"bar0", "1"}, {"bar1", "0.05"}});
    rfs->manager.transition(MICRO, "function0_right", "20");
    cs.render_microblock();
    rfs->manager.set("function1_left", "<function0_right>");
    rfs->manager.set("function1_right", "20");
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function0_right", "0");
    stage_macroblock(FileBlock("Memorizing the whole game tree before-hand takes up a ton of data, but no CPU."), 2);
    bcs->change_title(MICRO, "Lookup Table");
    bcs->manager.transition(MICRO, {{"bar0", "0.05"}, {"bar1", "1"}});
    cs.render_microblock();
    cs.render_microblock();

    rfs->manager.transition(MICRO, "function0_right", "<center_x>");
    stage_macroblock(FileBlock("We can balance these extremes by memorizing openings and searching at the end, demanding just a little memory and just a little compute."), 2);
    bcs->change_title(MICRO, "Memorize Openings, Search Endgames");
    bcs->manager.transition(MICRO, {{"bar0", "0.2"}, {"bar1", "0.2"}});
    cs.render_microblock();
    cs.fade_subscene(MICRO, "rfs", 0);
    cs.render_microblock();
    cs.remove_subscene("rfs");

    shared_ptr<C4Scene> steady_state = make_shared<C4Scene>("44444221", .6, .6);
    steady_state->set_fast_mode(true);
    cs.add_scene_fade_in(MICRO, steady_state, "steady_state", .5, .5, 1, true);
    stage_macroblock(FileBlock("And finally, we can use pattern recognition to compress the tree upfront, demanding even less memory and compute,"), 2);
    bcs->change_title(MICRO, "Steady State Diagram Compression");
    cs.render_microblock();
    steady_state->set_annotations_from_steadystate(MICRO);
    bcs->manager.transition(MICRO, {{"bar0", "0.05"}, {"bar1", "0.05"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("at the behest of a lot of resources used in upfront data compression, and pattern identification."), 2);
    bcs->add_bar(MICRO, "Preparation");
    cs.render_microblock();
    bcs->manager.transition(MICRO, "bar2", "2");
    cs.render_microblock();

    stage_macroblock(FileBlock("In each case, we're just making trade-offs between different resources."), 2);
    cs.render_microblock();
    bcs->manager.transition(MICRO, {
        {"bar0", "{t} 1.38 * sin .4 * .5 +"},
        {"bar1", "{t} 2 * sin .4 * .5 +"},
        {"bar2", "{t} 1.77 * sin .4 * .5 +"},
    });
    cs.render_microblock();

    cs.fade_subscene(MICRO, "bcs", 0);
    stage_macroblock(FileBlock("Want to use less compute? That comes at the cost of other resources!"), 1);
    bcs->manager.transition(MICRO, "bar0", "0.05");
    cs.render_microblock();
    bcs->manager.transition(MICRO, {{"bar1", ".5"}, {"bar2", ".3"}});
    cs.render_microblock();
    cs.remove_subscene("bcs");

    stage_macroblock(FileBlock("Such is the nature of games like this- the rules are so simple that a child can follow them,"), 1);
    cs.render_microblock();

    // TODO wolfram meme?
    stage_macroblock(FileBlock("but it seems like there's just no way of cheating around the computational irreducibility that they yield."), 1);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes();
}

void ideas(CompositeScene& cs, shared_ptr<C4GraphScene> weakc4) {
    stage_macroblock(FileBlock("This video isn't about connect four. It's not entirely about computer science, either."), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("It's about how systems yield complexity that can't be expressed in simpler terms."), 1);
    cs.render_microblock();

    shared_ptr<Graph> g = weakc4->graph;
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("", .5, .5);
    cs.add_scene_fade_in(MICRO, weakc4, "weakc4");
    cs.add_scene_fade_in(MICRO, c4s, "c4s", .25, .25);
    string variation = "4444422226666";
    stage_macroblock(FileBlock("Just knowing the rules doesn't mean you can win without somehow relying heavily on computation, raw memorization, or _something_..."), variation.size());
    C4Board b(FULL, "");
    for(char c : variation) {
        string move(1, c);
        c4s->play(move);
        b.play_piece(move[0] - '0');

        flood_fill_edges_to_highlight(g, b, weakc4);

        cs.render_microblock();
    }
    stage_macroblock(FileBlock("you can't just 'intuitively realize' that the candlesticks opening is a naturally arising strategy."), 1);
    cs.render_microblock();
    shared_ptr<BouncingBallScene> bbs = make_shared<BouncingBallScene>(10000, 100, 100);
    cs.remove_all_subscenes();
    cs.add_scene(bbs, "bbs");
    stage_macroblock(CompositeBlock(FileBlock("Just knowing the laws of particle interactions doesn't mean you can predict the weather,"), FileBlock("and it doesn't tell you that there's such a thing as a cloud.")), 4);
    bbs->manager.transition(MACRO, "zoom", "-5");
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<Mp4Scene> cumulonimbus = make_shared<Mp4Scene>(vector<string>{"cumulonimbus"}, 8, 8);
    cumulonimbus->manager.transition(MICRO, {{"width", "1"}, {"height", "1"}});
    cs.add_scene_fade_in(MICRO, cumulonimbus, "cumulonimbus");
    cs.render_microblock();
    cs.remove_subscene("bbs");
    cs.render_microblock();

    stage_macroblock(FileBlock("But that doesn't stop us from discussing candlesticks openings or clouds."), 1);
    //TODO footage from the textbook
    cs.render_microblock();
    stage_macroblock(FileBlock("Maybe we can't concisely explain them from first principles, but that doesn't make them any less real."), 1);
    cs.render_microblock();
    Mp4Scene weatherman_clip(vector<string>{"weatherman_clip"});
    stage_macroblock(FileBlock("Clouds are best described not in the language of particle interations,"), 1);
    weatherman_clip.render_microblock();
    stage_macroblock(FileBlock("but in the language of cold fronts, dew points, and vapor pressure- a whole new philosophy spawned out of thin air, far abstracted from the rules."), 1);
    weatherman_clip.render_microblock();

    cs.remove_subscene("weatherman_clip");
    cs.remove_subscene("cumulonimbus");
    stage_macroblock(FileBlock("Connect 4 shows us a glimpse of that same emergent substance."), 1);
    cs.render_microblock();
    shared_ptr<Mp4Scene> galaxy_clip = make_shared<Mp4Scene>(vector<string>{"galaxy"});
    cs.add_scene_fade_in(MICRO, galaxy_clip, "galaxy_clip");
    stage_macroblock(FileBlock("It shows us that complexity and mystery aren't unique to those laws which gave us stars and galaxies, benzene rings and toothpaste."), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("but are rather an inevitability more fundamental than our particular reality."), 1);
    cs.render_microblock();

    shared_ptr<MandelbrotScene> ms = make_shared<MandelbrotScene>();
    cs.add_scene_fade_in(MICRO, ms, "mandelbrot", .5, .5, .5, true);
    ms->manager.begin_timer("zoom");
    ms->manager.set("seed_c_r", "0.743643887037151");
    ms->manager.set("seed_c_i", "0.131825904205330");
    stage_macroblock(FileBlock("Equivalent magic arises from even the most humble systems of iterated rules, and this game is no exception."), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("Complex enough to invoke that same spark of emergent fertility in full force,"), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("but simple enough that with modern computers, I can show you this graph and say,"), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("'Here is its shape. This is connect 4.'"), 1);
    cs.render_microblock();

    //TODO 2swap logo and 3s pause
}

void anki(CompositeScene& cs) {
    shared_ptr<Mp4Scene> anki_clip = make_shared<Mp4Scene>(vector<string>{"anki"});
    cs.add_scene_fade_in(MICRO, anki_clip, "anki_clip");
    stage_macroblock(FileBlock("Alright, so I claimed to have made a connect 4 solution that's human-memorizable,"), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_all_subscenes_except("anki_clip");
    stage_macroblock(FileBlock("but it's time to put that to the test-"), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("I made a flash card deck using the spaced-repetition memorization app Anki, which I will attempt to learn!"), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("If you want to try too, you can download the deck in the description!"), 1);
    cs.render_microblock();

    shared_ptr<Mp4Scene> site_clip = make_shared<Mp4Scene>(vector<string>{"site_clip"});
    cs.add_scene_fade_in(MICRO, site_clip, "site_clip");
    stage_macroblock(FileBlock("I've also included a website where you can play against the weak solution,"), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("site_clip");
    stage_macroblock(FileBlock("and explore the tree as you traverse it."), 1);
    cs.render_microblock();
    stage_macroblock(FileBlock("Good luck winning... you'll need it!"), 1);
    cs.render_microblock();

    shared_ptr<PngScene> explanation = make_shared<PngScene>("explanation_page", 1.01, 1.01*2.85037 * VIDEO_WIDTH / VIDEO_HEIGHT);
    cs.add_scene_fade_in(MICRO, explanation, "explanation", .5, 2);
    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("explanation");

    cs.slide_subscene(MICRO, "explanation", 0, -1);
    stage_macroblock(FileBlock("The site includes a lot of technical details which I glazed over, so be sure to check it out!"), 1);
    cs.render_microblock();
}

void render_video() {
    VIDEO_BACKGROUND_COLOR = 0xff000022;
    CompositeScene cs;
    intro(cs);
    build_graph(cs);
    explanation(cs);
    trees(cs);
    patterned(cs);
    trimmed_solution(cs);
    shared_ptr<C4GraphScene> weakc4 = hardest_openings(cs);
    solution_types(cs);
    ideas(cs, weakc4);
    anki(cs);
}
