#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/Mp4Scene.h"
#include "../Scenes/Common/CompositeScene.h"
#include <memory>
#include "../Scenes/Math/RubiksGraphScene.h"
#include "../Scenes/Physics/RopeScene.h"
#include "../Core/State/StateTester.h"

/* SCRIPT
What does a rubik's cube, parallel parking, and rotating a broken sattalite have in common ?
They all make use of a special kind or sequence of moves, called a commutator.


Picture this : you are trying to hang a painting, but you want to make it hard for yourself,
you want it to be so that if you remove one of those two pins, the painting immediatly falls down.
You think it's easy ? Alright, try with three pins now, the painting should fall if either of the pins is removed.
Do you think it's possible with 4 pins ? With 5 ? Any number ?
Indeed it is, and the general solution involves nesting commutators !



*cube statespace graph*
Here is the solved cube,
let's make one random turn on it, we are now on one those twelve nodes,
now a second turn,
a third,
and one more...
At a depth of 10(change that to be correct) moves, we got the entire statespace of a 2x2 rubik's cube.
It has 3.674.160 nodes !
Now let's try the same thing with a 3x3.
Here is the solved cube.
Now one move deep,
two,
three,
TODO QTM vs HTM
as you can see, this graph grows MUCH faster than the 2x2 one, and it is in fact, wayyyy too big to completely render at once.
It actually has more than 43 quintillion nodes, that more than 10 to the power of 19 (huuuuh not sure, need to check)
The general formula for computing the number of nodes of the statespace graph of an nxn rubik's cube is the following (kindly provided by covoisinage :sunglasses:)
And as you can see, it has the growth rate of an exponential.
Even without having the full graph at hand, we can still see some interesting pattenrs ans symetries.
There are also a lot of loops, like this one (sexy move), this one (some random loop), or this trivial one (U4 from any state).
In fact, loops are extremely common, and there is a reason for that. The formal explanation is beyong the scope of this video, but it comes from group theory, and Lagrange's theorem states that : since there is a finite number of nodes in this graph, then, starting at any node, and reaping any sequence of move enough times will inevitably lead back to your starting point.

//At some point, take like 30 seconds to just travel through the graph and show interesting places, and patterns//


*subgroup inclusion, 2x2 into 3x3*

-----------------------------------------------------------------------------------
Graph state space
Graph state space where only shortest paths to the root are included

loops in the graphs, talk about the order of a sequence

U' R U R' F' U F U' sledge (3 comms in a row)
sune, and diagramm commutative thing
Abelianized group of 3x3 (maybe to hard)

Cube itself
Ring diagram
2-gen subgroup, color lattice points based on group element?
^ draw commutators on that subgroup
Interpolating between puzzles
15 puzzle
Big grid of permutable items

Other things with commutators
painting puzzle
parallel parking
quintic proof

corner and edge graphs (homomorphism with 2x2)
Bandage cube graph?
Animating elements of rubiks cube group as permutations which rip pieces out and put them back in
Group actions "rolling" graph along an automorphism in 3-space
(use moves to apply them on the whole graph, making it land on itself but symmetrically)
6d matrix of move distances
drawing shortest path to origin
v it would probably look like some sort of random walk in the below space

Make points on graph closer to each other if they have more pieces in the same spots
^ that neatly maps onto drawing commutators in a "geometric" way
^we could also tie that into the commutator on a lattice idea
Algorithms in each step of the cube and what their graphs/groups look like


Story:
Why not just draw a state space graph of the cube?
It's too big, but...
The cube has symmetry!
Bandage cube?
Symmetry means that we don't have to concern ourselves with all of the specifics
can abstract out general patterns
Visualize that symmetry somehow
Symmetric, but not abelian
What are the consequences?
r u r' u' doesn't come back to solved!!! (not abelian)
this means we can make commutators
what do those do??


use big formula to show that space state is too big
state space graph :
separate into edges orientation/perm, same for corners, show them all at the same time with a muving cube
for the 2x2, probably full graph (3.6 million nodes)
*/

/*Intro
Let's play a game.
Here's a rope, the goal is to tie it around one pin, 
in a way that if you remove the pin, the rope immediatly falls down.
Trivial, you'd say...
Ok now let's try with two pins, the rope should fall if either of the pins is removed.
The same solution doesn't work, so let's try something else...
This doesn't work either, let's try something else again...
Here's a solution, we first go around the first pin, then the second, then we go back around the first pin in the opposite direction, and finally we go back around the second pin in the opposite direction.
This is called a commutator, and with this path, if you remove either of the pins, the rope completely unties itself and falls down.
*/

void intro_rope_old(){
    RopeScene rs("io_in/loop_trivial", vec2(1, 1));
    RopeScene rs2("io_in/loop_trivial", vec2(1, 1));
    RopeScene rs3("io_in/loop_comm", vec2(1, 1));
    RopeScene rs4("io_in/loop_comm", vec2(1, 1));
    stage_macroblock(SilenceBlock(10), 6);
    
    rs.render_microblock();

    rs.add_pin(vec2(0.5,0.4));
    rs.render_microblock();

    rs.remove_pin(0);
    rs.render_microblock();

    rs2.add_pin(vec2(0.4,0.4));
    rs2.add_pin(vec2(0.6,0.4));
    rs2.render_microblock();

    rs2.remove_pin(0);
    rs2.render_microblock();

    rs2.add_pin(vec2(0.4,0.4));
    rs2.remove_pin(0);
    rs2.render_microblock();


    stage_macroblock(SilenceBlock(10), 4);
    
    rs3.add_pin(vec2(0.25, 0.3));
    rs3.add_pin(vec2(0.75, 0.3));
    rs3.render_microblock();

    rs3.remove_pin(0);
    rs3.render_microblock();

    rs4.add_pin(vec2(0.25, 0.3));
    rs4.add_pin(vec2(0.75, 0.3));
    rs4.render_microblock();

    rs4.remove_pin(1);
    rs4.render_microblock();
}

void intro_rope(){
    RopeScene rs("io_in/loop_trivial", vec2(1, 1));
    stage_macroblock(FileBlock("Let's play a game. Here's a rope,"));
    rs.render_microblock();

    stage_macroblock(FileBlock("The goal is to tie this rope around this pin, but there's an additionnal rule."));
    rs.add_pin(vec2(0.5,0.4));
    rs.render_microblock();

    stage_macroblock(FileBlock("If you remove the pin, the rope should be completely free."));
    rs.remove_pin(0);
    rs.render_microblock();

    stage_macroblock(FileBlock("Trivial, you may say..."));
    rs.render_microblock();

    RopeScene rs2("io_in/loop_trivial", vec2(1, 1));
    stage_macroblock(FileBlock("Let's make it a bit harder with two pins this time."));
    rs2.add_pin(vec2(0.2,0.4));
    rs2.add_pin(vec2(0.8,0.4));
    rs2.render_microblock();

    stage_macroblock(FileBlock("The rope should be free if either of the pins is removed."));
    rs2.remove_pin(0);
    rs2.render_microblock();

    stage_macroblock(FileBlock("The same solution doesn't work, so let's try something else..."));
    rs2.render_microblock();

    RopeScene rs3("io_in/loop_comm", vec2(1, 1));
    stage_macroblock(FileBlock("Here's a solution, we first go around the first pin, then the second, then we go back around the first pin in the opposite direction, and finally we go back around the second pin in the opposite direction."));
    rs3.add_pin(vec2(0.25, 0.3));
    rs3.add_pin(vec2(0.75, 0.3));
    rs3.render_microblock();
    stage_macroblock(SilenceBlock(3));
    rs3.render_microblock();

    stage_macroblock(FileBlock("In this situation, removing either of the pins, breaks the rope completely free,"));
    rs3.remove_pin(0);
    rs3.render_microblock();

    stage_macroblock(FileBlock("because around the remaining pin, we did a loop, and then undid it."));
    rs3.render_microblock();

    LatexScene ls("[A,B]=ABA^{-1}B^{-1}", vec2(1, 0.5));
    stage_macroblock(FileBlock("This is called a commutator, and we write it like this for the two pins, A and B"));
    ls.render_microblock();

    const string triple_com = "[A,B,C]";
    ls.begin_latex_transition(MICRO, triple_com);
    stage_macroblock(FileBlock("But what does a triple commutator look like ?"));
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("How would you need to tie the rope around those three pins to make it wall whenever we remove one of the pins ?"));
    ls.render_microblock();//TODO make a failed solution and remove one pin, something appears on screen showing we failed, idk

    stage_macroblock(FileBlock("Let's look at it algebraically."));
    ls.render_microblock();// TODO ajouter les ropes en plus des latex

    stage_macroblock(FileBlock("Looking back at the 2 pins case, we had this solution"));
    const string simple_commutator = "[A,B]=ABA^{-1}B^{-1}";
    ls.begin_latex_transition(MICRO, simple_commutator);
    ls.render_microblock();

    stage_macroblock(FileBlock("And in this situation, removing say pin A, leads to having only B clockwise and then B counterclockwise."));
    const string without_A = "[\\cdot,B]=BB^{-1}";
    ls.begin_latex_transition(MICRO, without_A);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("It also works by removing pin B, this was the goal of the game."));
    const string without_B = "[A,\\cdot]=AA^{-1}";
    ls.begin_latex_transition(MICRO, without_B);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("Ok, let's now add another pin C to this solution to make it work with three pins"));
    const string aled = "ABCA^{-1}B^{-1}C^{-1}";
    ls.begin_latex_transition(MICRO, aled);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("And now, if we remove pin A, we have this"));
    const string without_A2 = "BCB^{-1}C^{-1}";
    ls.begin_latex_transition(MICRO, without_A2);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("This solution doesn't work, because the remaining part isn't trivial, and the rope is still tied around the remaining pins."));
    ls.render_microblock();

    stage_macroblock(FileBlock("To make it fit our rules, we're gonna treat pins A and B as one entity, so that we reduce to the 2 pins case, with the block AB and the pin C"));
    const string block_AB = "[AB,C]=ABC(AB)^{-1}C^{-1}";
    ls.begin_latex_transition(MICRO, block_AB);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("and then, recall the solution for AB is the commutator [A,B]"));
    const string block_AB2 = "[[A,B],C]=[A,B]C[A,B]^{-1}C^{-1}";
    ls.begin_latex_transition(MICRO, block_AB2);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("and the full solution extends like this"));
    const string full_solution = "ABA^{-1}B^{-1}CBA^{-1}B^{-1}A^{-1}C^{-1}";
    ls.begin_latex_transition(MICRO, full_solution);
    ls.render_microblock();
    ls.render_microblock(); //TODO indiquer chaque lettre pendant que la rope fait le chemn de cette lettre, puis montrer que si on enlève un pin, le reste se défait.

    stage_macroblock(FileBlock("Of course, we could nest more commutators to make it work with any number of pins"));
    const string nested_commutators = "[[[A,B],C],D\\dots]";
    ls.begin_latex_transition(MICRO, nested_commutators);
    ls.render_microblock();
    ls.render_microblock();

    stage_macroblock(FileBlock("Here is the solution with 4 pins"));
    ls.render_microblock(); // TODO
}

void loop_4(){
    RopeScene rs("io_in/loop_4_comm", vec2(1, 1));
    stage_macroblock(SilenceBlock(10));
    
    rs.add_pin(vec2(0.25, 0.3));
    rs.add_pin(vec2(0.75, 0.3));
    rs.add_pin(vec2(0.25, 0.7));
    rs.add_pin(vec2(0.75, 0.7));
    rs.render_microblock();


}

void test_latex(){
    string latex_formula = "\\frac{7!\\times 3^6\\times 24!^{\\frac{n^2-2n-3\\times (n\\, mod\\, 2)}{4}}\\times (24\\times 12!\\times 2^{10})^{n\\, mod \\, 2}}{4!^{6\\times\\frac{(n-2)^2-n\\, mod\\, 2}{4}}}";
    string latex_oui = "OUI";
    LatexScene ls(latex_formula, 1);
    stage_macroblock(SilenceBlock(1), 1);
    ls.render_microblock();
}

void cube_corner_in_center(){
    RubiksScene rs;
    stage_macroblock(SilenceBlock(1), 1);

    quat yaw_quat = quat(cos(0.125 * M_PI), 0, sin(0.125 * M_PI), 0);
    quat pitch_quat = quat(cos(-0.098 * M_PI), sin(-0.098 * M_PI), 0, 0);
    quat combined_quat = pitch_quat * yaw_quat;

    rs.manager.transition(MACRO, {
        {"q1", to_string(combined_quat.u)},
        {"qi", to_string(combined_quat.i)},
        {"qj", to_string(combined_quat.j)},
        {"qk", to_string(combined_quat.k)},
        {"d", "1.4"},
        {"fov", "0.25"}
    });

    // d = 1.4, fov = 0.25

    
    rs.render_microblock();
    // open_ui(rs);


    stage_macroblock(SilenceBlock(5), 3);
    
    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("B'");
    rs.render_microblock();

    // rs.exec_move_from_slice("B2");
    // rs.render_microblock();

    rs.exec_move_from_slice("U");
    rs.render_microblock();

    // rs.exec_move_from_slice("R'");
    // rs.render_microblock();

    // rs.exec_move_from_slice("D");
    // rs.render_microblock();

    // rs.exec_move_from_slice("R");
    // rs.render_microblock();

    // rs.exec_move_from_slice("U'");
    // rs.render_microblock();

    // rs.exec_move_from_slice("R'");
    // rs.render_microblock();

    // rs.exec_move_from_slice("D'");
    // rs.render_microblock();

    // get the hash of the cube after the T perm and print it
    double hash = rs.the_cube.get_hash(3);
    std::cout << "Hash of the cube after T perm: " << setprecision(10)<< hash << std::endl;






    // stage_macroblock(SilenceBlock(10), 1);
    // rs.render_microblock();
}

void test_voice(){
    RubiksScene rs;
    stage_macroblock(FileBlock("nothing"), 1);
    rs.render_microblock();
}


void intro(CompositeScene& cs){
    shared_ptr<RubiksScene> rs = make_shared<RubiksScene>();
    cs.add_scene(rs, "rs");

    stage_macroblock(SilenceBlock(2), 1);
    rs->manager.transition(MACRO, {
        {"q1", "0.5"},
        {"qi", "{t} sin"},
        {"qj", "{t} cos"},
        {"qk", "0"},
        {"d", "4"},
    });
    cs.render_microblock();

    
    stage_macroblock(FileBlock("nothing"), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("nothing again test"), 1);
    cs.render_microblock();


    
}

void graph_one(){
    RubiksGraphScene rgs;
    rgs.manager.set({
        {"physics_multiplier", "40"},
        {"decay", ".8"},
        {"dimensions", "3"},
        {"d", "50"},
        {"qi", "{t} 5 * sin .2 *"},
        {"qj", "{t} 5 * cos .2 *"},
        {"cube_size", "3"},
    });

    stage_macroblock(FileBlock("Now let's do the same with a 3x3 !"), 1);
    rgs.add_cube("", true, false);
    rgs.render_microblock();

    stage_macroblock(FileBlock("let's make a random turn, we are now on one of those twelve nodes"), 1);
    rgs.add_children({"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"}, true, false, false);
    rgs.render_microblock();

    stage_macroblock(FileBlock("now a second turn"), 1);
    rgs.add_children({"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"}, true, false, false);
    rgs.render_microblock();

    stage_macroblock(FileBlock("and a third"), 1);
    rgs.add_children({"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"}, true, false, false);
    rgs.render_microblock();
    
}

void test_rope(){
    RopeScene rs("io_in/loop_example_0", vec2(1, 1));
    stage_macroblock(SilenceBlock(10), 1);
    
    rs.render_microblock();
}

void t_perm(){
    RubiksScene rs;
    stage_macroblock(SilenceBlock(1), 1);
    rs.render_microblock();

    stage_macroblock(SilenceBlock(10), 14);
    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("U");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("U'");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("F");
    rs.render_microblock();

    rs.exec_move_from_slice("R2");
    rs.render_microblock();

    rs.exec_move_from_slice("U'");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("U'");
    rs.render_microblock();

    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("U");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("F'");
    rs.render_microblock();
}


void render_video() {
    // CompositeScene cs;
    // intro(cs);
    // test_rope();
    // cube_corner_in_center();
    // t_perm();
    // intro_rope();
    loop_4();
}
