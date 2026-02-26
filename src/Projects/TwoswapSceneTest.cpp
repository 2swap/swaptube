#include "../Scenes/Common/TwoswapScene.h"

void render_video() {
    TwoswapScene ts;
    stage_macroblock(SilenceBlock(3), 3);
    ts.manager.transition(MICRO, "2swap_effect_completion", "1");
    ts.render_microblock();
    ts.manager.transition(MICRO, "6884_effect_completion", "1");
    ts.render_microblock();
    ts.render_microblock();
}
