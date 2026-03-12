#include "../Scenes/Media/CodeScene.h"

void render_video() {
    CodeScene cs;
    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
}
