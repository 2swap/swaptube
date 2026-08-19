#include "../Scenes/Math/VolumetricFractalScene.h"

void render_video() {
    VolumetricScene ms;
    stage_macroblock(SilenceBlock(3), 3);
    ms.manager.transition(MICRO, {{"qj", "1"}});
    ms.render_microblock();
    ms.manager.transition(MICRO, {{"qj", "0"}, {"qi", "-1"}});
    ms.render_microblock();
    ms.manager.transition(MICRO, {{"d", "2"}});
    ms.render_microblock();
    
    /*stage_macroblock(SilenceBlock(2), 2);
    vs.manager.transition(MICRO, { {"qk", "1"}, {"d", "4"}, {"power", "2"}});
    vs.render_microblock();
    vs.manager.transition(MICRO, {{"d", "0.02"}, {"qj", "-1"}, {"x", "-1.77"}});
    vs.render_microblock();*/
}
