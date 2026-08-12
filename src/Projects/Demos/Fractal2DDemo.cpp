#include "../Scenes/Math/Fractal2DScene.h"

void render_video() {
    Fractal2DScene fs;
    stage_macroblock(SilenceBlock(2), 1);
    /*
    fs.manager.set("max_iterations", "200");
    fs.manager.set(fs.makeFractalStateSet({0, 0, 2, 0, -0.85, 0.15}, "O", MANDELBROT_XSET));
    fs.manager.set(fs.makeFractalStateSet({2, 0, 0, 0, 0, 0}, "x", MANDELBROT_XSET));
    fs.manager.set(fs.makeFractalStateSet({0, 2, 0, 0, 0, 0}, "y", MANDELBROT_XSET));
    fs.render_microblock();
    */
    fs.manager.set({{"fractal_mode", to_string(MANDELBROT_XSET)}, {"burning", to_string(0b00000000)}});
    fs.manager.set(fs.makeFractalStateSet({0, 0, 2, 0, 0, 0}, "O", MANDELBROT_XSET));
    fs.manager.set(fs.makeFractalStateSet({0, 0, 0, 0, 2, 0}, "X", MANDELBROT_XSET));
    fs.manager.set(fs.makeFractalStateSet({0, 0, 0, 0, 0, 2}, "Y", MANDELBROT_XSET));
    stage_macroblock(SilenceBlock(2), 2);

    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 0, 2, 0, 0, 0}, "O", MANDELBROT_XSET));
    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 0, 2, 0, 0, 0}, "X", MANDELBROT_XSET));
    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 0, 0, 2, 0, 0}, "Y", MANDELBROT_XSET));
    fs.manager.transition(MICRO, {{"sub_dimensions_x", "15"}, {"sub_dimensions_y", "15"}});
    fs.render_microblock();

    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 0, 2, 0, 0, 0}, "O", MANDELBROT_XSET));
    fs.manager.transition(MICRO, fs.makeFractalStateSet({2, 0, 0, 0, 0, 0}, "x", MANDELBROT_XSET));
    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 2, 0, 0, 0, 0}, "y", MANDELBROT_XSET));
    fs.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 0, 2, 0, -1, 0.5}, "O", MANDELBROT_XSET));
    fs.render_microblock();
}