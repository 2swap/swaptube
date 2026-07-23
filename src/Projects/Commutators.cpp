#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"
#include "../Scenes/Media/LatexScene.h"



void test_latex(){
    string latex_formula = "\\frac{7!\\times 3^6\\times 24!^{\\frac{n^2-2n-3\\times (n\\, mod\\, 2)}{4}}\\times (24\\times 12!\\times 2^{10})^{n\\, mod \\, 2}}{4!^{6\\times\\frac{(n-2)^2-n\\, mod\\, 2}{4}}}";
    string latex_oui = "OUI";
    LatexScene ls(latex_formula, 1);
    stage_macroblock(SilenceBlock(1), 1);
    ls.render_microblock();
}

void t_perm(){
    RubiksScene rs("R");
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
    double hash = rs.the_cube->get_hash();
    std::cout << "Hash of the cube after T perm: " << setprecision(10)<< hash << std::endl;




    stage_macroblock(SilenceBlock(5), 1);
    rs.render_microblock();
}



void render_video() {
    t_perm();
}
