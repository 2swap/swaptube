#include "../Scenes/Media/StateSliderScene.h"

void render_video() {
    StateSliderScene sss("{microblock_fraction}", "\\text{micro fraction}", 0, 1);
    stage_macroblock(CompositeBlock(SilenceBlock(5), SilenceBlock(1)), 5);
    for(int i = 0; i < 5; i++) sss.render_microblock();
    stage_macroblock(SilenceBlock(1), 2);
    for(int i = 0; i < 2; i++) sss.render_microblock();
}

