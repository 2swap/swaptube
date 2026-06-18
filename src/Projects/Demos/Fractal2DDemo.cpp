#include "../Scenes/Math/Fractal2DScene.h"

void render_video() {
    Fractal2DScene fs;
    fs.manager.set({{"fractal_mode", to_string(COMPLEX_C_COEFF_POLY)}, {"burning", to_string(0b11111111)}});
    fs.manager.set(fs.makeFractalStateSet({0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "O", COMPLEX_C_COEFF_POLY));
    fs.manager.set(fs.makeFractalStateSet({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0}, "X", COMPLEX_C_COEFF_POLY));
    fs.manager.set(fs.makeFractalStateSet({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}, "Y", COMPLEX_C_COEFF_POLY));

    stage_macroblock(SilenceBlock(2), 1);

    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 0, 1, 0, 0, 0, 3, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.6, -1.3}, "O", COMPLEX_C_COEFF_POLY));
    fs.manager.transition(MICRO, fs.makeFractalStateSet({1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "X", COMPLEX_C_COEFF_POLY));
    fs.manager.transition(MICRO, fs.makeFractalStateSet({0, 2, 0, 0.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "Y", COMPLEX_C_COEFF_POLY));

    fs.render_microblock();
}
