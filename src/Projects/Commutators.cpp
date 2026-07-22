#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"

void render_video() {

    RubiksScene rs;
    stage_macroblock(SilenceBlock(1), 1);

    rs.manager.transition(MACRO, {
        {"q1", "{t}"},
        {"qi", "{t} sin"},
        {"qj", "{t} cos"},
        {"qk", "0"},
        {"d", "4"},
    });
    rs.render_microblock();


    stage_macroblock(SilenceBlock(10), 6);
    rs.exec_move_from_slice('F', 0);
    rs.render_microblock();

    rs.exec_move_from_slice('R', 0);
    rs.render_microblock();

    rs.exec_move_from_slice('U', 1);
    rs.render_microblock();

    rs.exec_move_from_slice('D', 2);
    rs.render_microblock();

    rs.exec_move_from_slice('L', 1);
    rs.render_microblock();

    rs.exec_move_from_slice('B', 0);
    rs.render_microblock();
}


    
    
    