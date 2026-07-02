#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"

void render_video() {
    test_rubiks();

    RubiksScene rs;
    stage_macroblock(SilenceBlock(1), 1);

    rs.manager.transition(MACRO, {
        {"q1", "{t} 2 / sin"},
        {"qi", "{t} 2 * sin .6 *"},
        {"qj", "{t} 2 / cos"},
        {"qk", "0"},
        {"d", "4"},
    });
    rs.render_microblock();

    stage_macroblock(SilenceBlock(10), 1);

    rs.render_microblock();


}


    
    
    