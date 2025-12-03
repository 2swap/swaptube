#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../DataObjects/Connect4/TreeValidator.cpp"

void render_video() {
    C4Scene c4("444");
    c4.stage_macroblock(FileBlock("I found a profoundly better way of playing perfectly optimal connect 4."), 1);
    c4.render_microblock();
    return;
    c4.stage_macroblock(FileBlock("You see, connect 4 has been solved since 1988."), 1);
    c4.stage_macroblock(FileBlock("We learned then that with perfect play, the first player, Red, can always guarantee a win by playing optimally."), 1);
    c4.stage_macroblock(FileBlock("But how do we actually achieve such optimal play?"), 1);
    c4.stage_macroblock(FileBlock("What's the trick? What's the rule to follow to play perfectly?"), 1);
    c4.stage_macroblock(FileBlock("Clever methods were concocted to optimize traditional brute force search for the computers of the era,"), 1);
    c4.stage_macroblock(FileBlock("meaning we didn't discover any quick trick to play right- we just searched through all possible variations of all possible variations until we found that it is always possible for Red to force a win."), 1);
    c4.stage_macroblock(FileBlock("But it's not the 1980s anymore."), 1);
    c4.stage_macroblock(FileBlock("If you know my channel, you know that I love to illustrate the structure beneath complex systems."), 1);
    c4.stage_macroblock(FileBlock("In this video, I will present to you a complete solution of connect 4 so simple that it can be visualized entirely, and maybe, just maybe, even be memorized and used by a human."), 1);

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

    I found a profoundly simple strategy to play perfectly optimal chess.
    It turns out, if you label each piece, write the state of the board out in binary, plug it into this formula, take the sha256 hash of the result, the first few letters of the result are the optimal move in standard algebraic notation.
    There's just one problem... it's completely made up.
    There's no simple trick like that for games like chess.
    but how do we know?

    This video isn't about chess. It's not really about connect 4, either. 
    It's about systems' ability to yield complexity that cannot be expressed in simpler terms.
    It's about our language, and its insufficiency to perfectly describe the world around us.
    It's about emergent behavior- behavior not baked into the 'rules of the game', but arising from them.

    Just like connect 4, our world is a myriad of emergent objects built on a bedrock of simple rules.
    It may contain systems like double pendulums, which are fundamentally unpredictable despite being deterministic. But that doesn't stop us from navigating and discussing the world around us with our mere finite words and models.
    Connect 4 shows us a glimpse of that emergent substance, distilled in a system which we can play with and study.


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
