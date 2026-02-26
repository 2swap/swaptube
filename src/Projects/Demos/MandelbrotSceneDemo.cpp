#include "../Scenes/Math/MandelbrotScene.h"

void render_video() {
    MandelbrotScene ms;
    ms.manager.transition(MICRO, {{"pixel_param_z", "1"}, {"pixel_param_c", "0"}});
    stage_macroblock(SilenceBlock(2), 1);
    ms.render_microblock();
}
