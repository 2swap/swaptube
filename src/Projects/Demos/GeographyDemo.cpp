#include "../Scenes/Common/GeographyScene.h"

void render_video() {
    GeographyScene gs;

    stage_macroblock(SilenceBlock(1), 1);
    gs.manager.set({
        {"d", "10"},
    });
    gs.manager.transition(MICRO, {
        {"qi", ".2"},
        {"qj", "{t} 2 / sin"},
        {"qk", ".1"},
    });
    gs.render_microblock();
}
