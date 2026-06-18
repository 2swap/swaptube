#include "../Scenes/Math/VolumetricFractalScene.h"

void render_video() {
    VolumetricScene vs;
    stage_macroblock(SilenceBlock(2), 1);
    vs.manager.transition(MICRO, { {"qk", "1"}/*{"power", "8"}*/});
    vs.render_microblock();
}
