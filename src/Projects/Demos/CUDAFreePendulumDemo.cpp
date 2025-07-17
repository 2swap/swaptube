#include "../Scenes/Physics/PendulumScene.cpp"

void render_video() {
    PendulumScene ps({0.5, 1.2, 0, 0});
    ps.stage_macroblock(SilenceBlock(10), 1);
    ps.render_microblock();
}

