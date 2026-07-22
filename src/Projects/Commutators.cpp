#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"

void render_video() {
    stage_macroblock(FileBlock("What does a rubik's cube, parallel parking, solving quintic polynomials, orienting a broken satellite, and hanging a painting all have in common?"), 1);
    rs.render_microblock();

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


    stage_macroblock(SilenceBlock(5), 8);
    // rs.manager.transition(MACRO, {
    //     {"cube_size", "11"},
    // });
    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("U");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("D");
    rs.render_microblock();

    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("U'");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("D'");
    rs.render_microblock();




    stage_macroblock(SilenceBlock(5), 1);
    rs.render_microblock();
    



    
}
