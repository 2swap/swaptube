#include "../Scenes/Math/MandelbrotScene.h"
#include "../Core/State/StateTester.h"
#include "../Core/Smoketest.h"

void render_video() {
    MandelbrotScene ms;
    ms.manager.set("zoom", ".2");
    ms.manager.transition(MICRO, {
        {"pixel_param_z", "1"},
        {"pixel_param_c", "0"},
        {"seed_c_r", ".4"},
        {"seed_c_i", ".1"},
    });
    stage_macroblock(SilenceBlock(4), 2);
    ms.render_microblock();
    if (!is_smoketest())
        open_ui(ms);
    ms.manager.transition(MICRO, "zoom", "1");
    ms.render_microblock();
}
