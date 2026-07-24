#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/Mp4Scene.h"
#include "../Scenes/Common/CompositeScene.h"
#include <memory>
#include "../Scenes/Math/RubiksGraphScene.h"



void test_latex(){
    string latex_formula = "\\frac{7!\\times 3^6\\times 24!^{\\frac{n^2-2n-3\\times (n\\, mod\\, 2)}{4}}\\times (24\\times 12!\\times 2^{10})^{n\\, mod \\, 2}}{4!^{6\\times\\frac{(n-2)^2-n\\, mod\\, 2}{4}}}";
    string latex_oui = "OUI";
    LatexScene ls(latex_formula, 1);
    stage_macroblock(SilenceBlock(1), 1);
    ls.render_microblock();
}

void t_perm(){
    RubiksScene rs;
    stage_macroblock(SilenceBlock(1), 1);

    rs.manager.transition(MACRO, {
        {"q1", "0.5"},
        {"qi", "{t} sin"},
        {"qj", "{t} cos"},
        {"qk", "0"},
        {"d", "4"},
    });
    rs.render_microblock();


    stage_macroblock(SilenceBlock(5), 3);
    // rs.manager.transition(MACRO, {
    //     {"cube_size", "11"},
    // });
    rs.exec_move_from_slice("B");
    rs.render_microblock();

    rs.exec_move_from_slice("B'");
    rs.render_microblock();

    rs.exec_move_from_slice("B2");
    rs.render_microblock();

    // rs.exec_move_from_slice("U");
    // rs.render_microblock();

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
    double hash = rs.the_cube->get_hash(3);
    std::cout << "Hash of the cube after T perm: " << setprecision(10)<< hash << std::endl;






    stage_macroblock(SilenceBlock(5), 1);
    rs.render_microblock();
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
    rgs.add_cube("", true);
    rgs.render_microblock();

    stage_macroblock(FileBlock("let's make a random turn, we are now on one of those twelve nodes"), 1);
    rgs.add_children({"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"}, true);
    rgs.render_microblock();

    stage_macroblock(FileBlock("now a second turn"), 1);
    rgs.add_children({"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"}, true);
    rgs.render_microblock();

    stage_macroblock(FileBlock("and a third"), 1);
    rgs.add_children({"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"}, true);
    rgs.render_microblock();
    
}


void render_video() {
    // CompositeScene cs;
    // intro(cs);

    graph_one();
}
