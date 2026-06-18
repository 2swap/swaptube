#include "Fractal2DScene.h"

extern "C" void fractal_2D_render(
    const int width, const int height,
    float o[28], // origin parameters
    float x[28], // x coordinate parameter multipliers
    float y[28], // y coordinate parameter multipliers
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors
);

Fractal2DScene::Fractal2DScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"zrO", "0"}, {"ziO", "0"},
        {"a1rO", "0"}, {"a1iO", "0"}, {"ac1rO", "0"}, {"ac1iO", "0"}, {"x1rO", "0"}, {"x1iO", "0"},
        {"a2rO", "0"}, {"a2iO", "0"}, {"ac2rO", "0"}, {"ac2iO", "0"}, {"x2rO", "0"}, {"x2iO", "0"},
        {"a3rO", "0"}, {"a3iO", "0"}, {"ac3rO", "0"}, {"ac3iO", "0"}, {"x3rO", "0"}, {"x3iO", "0"},
        {"a4rO", "0"}, {"a4iO", "0"}, {"ac4rO", "0"}, {"ac4iO", "0"}, {"x4rO", "0"}, {"x4iO", "0"},
        {"crO", "0"}, {"ciO", "0"},
        {"zrX", "0"}, {"ziX", "0"},
        {"a1rX", "0"}, {"a1iX", "0"}, {"ac1rX", "0"}, {"ac1iX", "0"}, {"x1rX", "0"}, {"x1iX", "0"},
        {"a2rX", "0"}, {"a2iX", "0"}, {"ac2rX", "0"}, {"ac2iX", "0"}, {"x2rX", "0"}, {"x2iX", "0"},
        {"a3rX", "0"}, {"a3iX", "0"}, {"ac3rX", "0"}, {"ac3iX", "0"}, {"x3rX", "0"}, {"x3iX", "0"},
        {"a4rX", "0"}, {"a4iX", "0"}, {"ac4rX", "0"}, {"ac4iX", "0"}, {"x4rX", "0"}, {"x4iX", "0"},
        {"crX", "1"}, {"ciX", "0"},
        {"zrY", "0"}, {"ziY", "0"},
        {"a1rY", "0"}, {"a1iY", "0"}, {"ac1rY", "0"}, {"ac1iY", "0"}, {"x1rY", "0"}, {"x1iY", "0"},
        {"a2rY", "0"}, {"a2iY", "0"}, {"ac2rY", "0"}, {"ac2iY", "0"}, {"x2rY", "0"}, {"x2iY", "0"},
        {"a3rY", "0"}, {"a3iY", "0"}, {"ac3rY", "0"}, {"ac3iY", "0"}, {"x3rY", "0"}, {"x3iY", "0"},
        {"a4rY", "0"}, {"a4iY", "0"}, {"ac4rY", "0"}, {"ac4iY", "0"}, {"x4rY", "0"}, {"x4iY", "0"},
        {"crY", "0"}, {"ciY", "1"},
        {"burning", "0"},
        {"conj", "0"},
        {"fractal_mode", to_string(MANDELBROT_2)},
        {"max_iterations", "50"},
    });
};

// Creates a partial StateSet for the fractal scene based on the provided parameters
// mode (fractal mode) specifies proper size of input arrays
// MANDELBROT_2         : 4 parameters  {zr, zi, cr, ci}
// MANDELBROT_3         : 4 parameters  {zr, zi, cr, ci}
// MANDELBROT_POWER     : 5 parameters  {zr, zi, power, cr, ci}
// MANDELBROT_XSET      : 6 parameters  {zr, zi, xr, xi, cr, ci}
// REAL_COEFF_POLY      : 12 parameters {zr, zi, a1, x1 ... a4, x4, cr, ci}
// COMPLEX_COEFF_POLY   : 20 parameters {zr, zi, a1r, a1i, x1r, x1i, ... a4r, a4i, x4r, x4i, cr, ci}
// COMPLEX_C_COEFF_POLY : 28 parameters {zr, zi, a1r, a1i, ac1r, ac1i, x1r, x1i, ... a4r, a4i, ac4r, ac4i, x4r, x4i, cr, ci}
StateSet Fractal2DScene::makeFractalStateSet(vector<float> params, const string& identifier, const int mode) {
    switch(mode){
        case MANDELBROT_2:
        case MANDELBROT_3:
            return {
                {"zr" + identifier, to_string(params[0])}, {"zi" + identifier, to_string(params[1])}, 
                {"cr" + identifier, to_string(params[2])}, {"ci" + identifier, to_string(params[3])}
            };
        case MANDELBROT_POWER:
            return {
                {"zr" + identifier, to_string(params[0])}, {"zi" + identifier, to_string(params[1])},
                {"x1r" + identifier, to_string(params[2])},
                {"cr" + identifier, to_string(params[3])}, {"ci" + identifier, to_string(params[4])}
            };
        case MANDELBROT_XSET:
            return {
                {"zr" + identifier, to_string(params[0])}, {"zi" + identifier, to_string(params[1])},
                {"x1r" + identifier, to_string(params[2])}, {"x1i" + identifier, to_string(params[3])},
                {"cr" + identifier, to_string(params[4])}, {"ci" + identifier, to_string(params[5])}
            };
        case REAL_COEFF_POLY:
            return {
                {"zr" + identifier, to_string(params[0])}, {"zi" + identifier, to_string(params[1])},
                {"a1r" + identifier, to_string(params[2])}, {"x1r" + identifier, to_string(params[3])},
                {"a2r" + identifier, to_string(params[4])}, {"x2r" + identifier, to_string(params[5])},
                {"a3r" + identifier, to_string(params[6])}, {"x3r" + identifier, to_string(params[7])},
                {"a4r" + identifier, to_string(params[8])}, {"x4r" + identifier, to_string(params[9])},
                {"cr" + identifier, to_string(params[10])}, {"ci" + identifier, to_string(params[11])}
            };
        case COMPLEX_COEFF_POLY:
            return {
                {"zr" + identifier, to_string(params[0])}, {"zi" + identifier, to_string(params[1])},
                {"a1r" + identifier, to_string(params[2 ])}, {"a1i" + identifier, to_string(params[3 ])}, {"x1r" + identifier, to_string(params[4 ])}, {"x1i" + identifier, to_string(params[5 ])},
                {"a2r" + identifier, to_string(params[6 ])}, {"a2i" + identifier, to_string(params[7 ])}, {"x2r" + identifier, to_string(params[8 ])}, {"x2i" + identifier, to_string(params[9 ])},
                {"a3r" + identifier, to_string(params[10])}, {"a3i" + identifier, to_string(params[11])}, {"x3r" + identifier, to_string(params[12])}, {"x3i" + identifier, to_string(params[13])},
                {"a4r" + identifier, to_string(params[14])}, {"a4i" + identifier, to_string(params[15])}, {"x4r" + identifier, to_string(params[16])}, {"x4i" + identifier, to_string(params[17])},
                {"cr" + identifier, to_string(params[18])}, {"ci" + identifier, to_string(params[19])}
            };
        case COMPLEX_C_COEFF_POLY:
            return {
                {"zr" + identifier, to_string(params[0])}, {"zi" + identifier, to_string(params[1])},
                {"a1r" + identifier, to_string(params[2 ])}, {"a1i" + identifier, to_string(params[3 ])}, {"ac1r" + identifier, to_string(params[4 ])}, {"ac1i" + identifier, to_string(params[5 ])}, {"x1r" + identifier, to_string(params[6 ])}, {"x1i" + identifier, to_string(params[7 ])},
                {"a2r" + identifier, to_string(params[8 ])}, {"a2i" + identifier, to_string(params[9 ])}, {"ac2r" + identifier, to_string(params[10])}, {"ac2i" + identifier, to_string(params[11])}, {"x2r" + identifier, to_string(params[12])}, {"x2i" + identifier, to_string(params[13])},
                {"a3r" + identifier, to_string(params[14])}, {"a3i" + identifier, to_string(params[15])}, {"ac3r" + identifier, to_string(params[16])}, {"ac3i" + identifier, to_string(params[17])}, {"x3r" + identifier, to_string(params[18])}, {"x3i" + identifier, to_string(params[19])},
                {"a4r" + identifier, to_string(params[20])}, {"a4i" + identifier, to_string(params[21])}, {"ac4r" + identifier, to_string(params[22])}, {"ac4i" + identifier, to_string(params[23])}, {"x4r" + identifier, to_string(params[24])}, {"x4i" + identifier, to_string(params[25])},
                {"cr" + identifier, to_string(params[26])}, {"ci" + identifier, to_string(params[27])}
            };
    }

    return {{}};
}

const StateQuery Fractal2DScene::populate_state_query() const {
    return {
        "zrO", "ziO",
        "a1rO", "a1iO", "ac1rO", "ac1iO", "x1rO", "x1iO",
        "a2rO", "a2iO", "ac2rO", "ac2iO", "x2rO", "x2iO",
        "a3rO", "a3iO", "ac3rO", "ac3iO", "x3rO", "x3iO",
        "a4rO", "a4iO", "ac4rO", "ac4iO", "x4rO", "x4iO",
        "crO", "ciO",
        "zrX", "ziX",
        "a1rX", "a1iX", "ac1rX", "ac1iX", "x1rX", "x1iX",
        "a2rX", "a2iX", "ac2rX", "ac2iX", "x2rX", "x2iX",
        "a3rX", "a3iX", "ac3rX", "ac3iX", "x3rX", "x3iX",
        "a4rX", "a4iX", "ac4rX", "ac4iX", "x4rX", "x4iX",
        "crX", "ciX",
        "zrY", "ziY",
        "a1rY", "a1iY", "ac1rY", "ac1iY", "x1rY", "x1iY",
        "a2rY", "a2iY", "ac2rY", "ac2iY", "x2rY", "x2iY",
        "a3rY", "a3iY", "ac3rY", "ac3iY", "x3rY", "x3iY",
        "a4rY", "a4iY", "ac4rY", "ac4iY", "x4rY", "x4iY",
        "crY", "ciY",
        "burning", "conj",
        "fractal_mode",
        "max_iterations"};
}

void Fractal2DScene::populateParamArray(float* params, const string& identifier) {
    params[0] = state["zr" + identifier];
    params[1] = state["zi" + identifier];
    for(int i = 1; i <= 4; i++){
        params[2 + (6 * (i - 1)) + 0] = state["a" + to_string(i) + "r" + identifier];
        params[2 + (6 * (i - 1)) + 1] = state["a" + to_string(i) + "i" + identifier];
        params[2 + (6 * (i - 1)) + 2] = state["ac" + to_string(i) + "r" + identifier];
        params[2 + (6 * (i - 1)) + 3] = state["ac" + to_string(i) + "i" + identifier];
        params[2 + (6 * (i - 1)) + 4] = state["x" + to_string(i) + "r" + identifier];
        params[2 + (6 * (i - 1)) + 5] = state["x" + to_string(i) + "i" + identifier];
    }
    params[26] = state["cr" + identifier];
    params[27] = state["ci" + identifier];
}

void Fractal2DScene::populateAllArrays() {
    populateParamArray(origin_params, "O");
    populateParamArray(x_params, "X");
    populateParamArray(y_params, "Y");
}

void Fractal2DScene::draw(){
    populateAllArrays();
    fractal_2D_render(pix.w, pix.h,
        origin_params,
        x_params,
        y_params,
        state["burning"], state["conj"],
        state["fractal_mode"],
        state["max_iterations"],
        pix.pixels.data());
}
