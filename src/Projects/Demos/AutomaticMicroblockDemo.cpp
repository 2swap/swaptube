#include "../Scenes/Math/MandelbrotScene.h"

#include <vector>

void render_video() {
    MandelbrotScene scene;
    const std::vector<double> zooms = {0.15, -1.0, 0.4, -1.0, 0.75};

    stage_macroblock(SilenceBlock(1));
    for (const double zoom : zooms) {
        if (zoom < 0) continue;
        scene.manager.transition(MICRO, "zoom", std::to_string(zoom));
        scene.render_microblock();
    }

    stage_macroblock(SilenceBlock(1));
    scene.render_microblock();
}
