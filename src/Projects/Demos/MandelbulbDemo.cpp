#include "../Scenes/Math/MandelbulbScene.h"

void render_video() {
    MandelbulbScene ms;
    stage_macroblock(SilenceBlock(10), 3);
    ms.manager.transition(MICRO, "qj", "1");
    ms.render_microblock();
    ms.manager.transition(MICRO, "d", "1");
    ms.render_microblock();
    ms.manager.transition(MICRO, "max_mandelbulb_iterations", "50");
    ms.render_microblock();
}
