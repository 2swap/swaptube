#include "../Scenes/Math/MandelbrotScene.cpp"

void render_video() {
    MandelbrotScene ms;
    ms.manager.set("zoom", "{t}");
    ms.stage_macroblock(SilenceBlock(2), 1);
    ms.render_microblock();
}
