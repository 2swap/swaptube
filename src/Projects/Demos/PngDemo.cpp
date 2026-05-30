#include "../Scenes/Media/PngScene.h"

void render_video() {
    PngScene ps("../earth_tiny.png");
    stage_macroblock(SilenceBlock(2), 1);
    ps.render_microblock();
}
