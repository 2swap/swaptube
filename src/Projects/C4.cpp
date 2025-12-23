#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Connect4/C4Scene.cpp"
#include "../Scenes/Connect4/C4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void render_video() {

    CompositeScene cs;
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>("");
    //shared_ptr<PngScene> god1 = make_shared<PngScene>("God1", .3, .5);
    //shared_ptr<PngScene> god2 = make_shared<PngScene>("God2", .3, .5);

    FOR_REAL = true;
    cs.stage_macroblock(FileBlock("Suppose two omniscient gods play a game of Connect 4."), 2);
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, c4s, "c4");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The rules are simple:"), 1);
    c4s->play("433335526245");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Players take turns dropping discs in the columns,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and the first to make a line of 4 wins."), 1);
    c4s->play("6");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Diagonals count too."), 2);
    c4s->undo(1);
    cs.render_microblock();
    c4s->play("4");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What would happen?"), 1);
    c4s->undo(100);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("God 1, playing red, plays the first piece in the center column."), 1);
    c4s->play("4");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("God 2, playing yellow, promptly resigns."), 1);
    cs.render_microblock();

    Graph g;
    shared_ptr<C4GraphScene> c4gs = make_shared<C4GraphScene>(&g, false, "", FULL);
    c4gs->manager.set("dimensions", "2");
    cs.add_scene_fade_in(MICRO, c4gs, "c4gs");
    cs.fade_subscene(MICRO, "c4", 0.4);
    cs.stage_macroblock(FileBlock("You see, after analyzing every possible variation of every opening, God 2 realized there was no way of stopping God 1 from making a red line of 4."), 1);
    cs.render_microblock();


    return;
    cs.stage_macroblock(FileBlock("This was first discovered by computer scientists in 1988."), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("They used strategies similar to the one that God 2 used:"), 1);
    cs.stage_macroblock(FileBlock("they wrote computer programs to search all possible variations,"), 1);
    cs.stage_macroblock(FileBlock("showing that player 1 is guaranteed to win, if they play perfectly."), 1);
    cs.stage_macroblock(FileBlock("This was wonderful work by the computer scientists of the day,"), 1);
    cs.stage_macroblock(FileBlock("But it kind of leaves you wanting."), 1);
    cs.stage_macroblock(FileBlock("Ok, so player 1 wins, but _why_?"), 1);
    cs.stage_macroblock(FileBlock("A computer might be able to iterate over millions or billions of nodes to check this result,"), 1);
    cs.stage_macroblock(FileBlock("But what's left for us humans?"), 1);
    cs.stage_macroblock(FileBlock("What are the best openings?"), 1);
    cs.stage_macroblock(FileBlock("Is there some change of perspective that shows how us mere mortals could beat god 2?"), 1);
    cs.stage_macroblock(FileBlock(""), 1);
    cs.stage_macroblock(FileBlock(""), 1);
    cs.stage_macroblock(FileBlock(""), 1);
    cs.stage_macroblock(FileBlock(""), 1);
}

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
