#include "../Scenes/Math/GeodesicScene.h"

void render_video() {
    shared_ptr<GeodesicScene> gs = make_shared<GeodesicScene>();

    gs->manager.set({
        {"scrunch_x", "2"},
        {"scrunch_y", "2"},
        {"scrunch_z", "2"},
        {"amp", "0"},
        {"pov_qj", "1"},
    });

    gs->manager.set({
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(a) <scrunch_x> * sin (b) <scrunch_y> * sin (c) <scrunch_z> * sin + + <amp> *"},
    });
    gs->manager.transition(MACRO, {
        {"pov_q1", "-.5"},
        {"pov_qj", "1"},
    });
    stage_macroblock(SilenceBlock(2), 2);
    gs->manager.transition(MICRO, {
        {"amp", ".25"}
    });
    gs->render_microblock();
    gs->manager.transition(MICRO, {
        {"amp", "0"}
    });
    gs->render_microblock();
}
