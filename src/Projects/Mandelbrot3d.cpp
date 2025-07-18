#include "../Scenes/Math/Mandelbrot3dScene.cpp"
void render_video() {
    Mandelbrot3dScene ms;

    ms.stage_macroblock(SilenceBlock(15), 3);
    ms.render_microblock();
    ms.state_manager.transition(MICRO, {{"seed_x_r", "3"}});
    ms.render_microblock();
    ms.render_microblock();
}
