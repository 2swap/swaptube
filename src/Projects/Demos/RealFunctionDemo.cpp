#include "../Scenes/Math/RealFunctionScene.h"

void render_video() {
    RealFunctionScene scene;
    stage_macroblock(SilenceBlock(5), 5);
    scene.render_microblock();
    scene.render_microblock();
    scene.manager.transition(MICRO, "function", "(a)");
    scene.render_microblock();
    scene.render_microblock();
    scene.manager.transition(MICRO, "function", "(a) (a) *");
    scene.render_microblock();
}
