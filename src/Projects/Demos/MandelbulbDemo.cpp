#include "../Scenes/Math/MandelbulbScene.h"

void render_video() {
    MandelbulbScene ms;
    stage_macroblock(SilenceBlock(3), 3);
    ms.manager.transition(MICRO, {{"qj", "1"}});
    ms.render_microblock();
    ms.manager.transition(MICRO, {{"qj", "0"}, {"qi", "1"}});
    ms.render_microblock();
    ms.manager.transition(MICRO, {{"d", "0.02"}});
    ms.render_microblock();
    //ms.manager.transition(MICRO, {{"sdflerp", "1"}});
    //ms.render_microblock();
}
