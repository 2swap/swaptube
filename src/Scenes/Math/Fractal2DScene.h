#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/vec.h"

// Each set of parameters needs 3 sets of inputs: Origin, x-pixel, and y-pixel
// They function as n-dimensional basis vectors for a linear, matrix-like function of (R^2) -> (R^n), where n is the number of parameterizable parameters
enum fractalModes {
    MANDELBROT_2,
    MANDELBROT_3,
    MANDELBROT_POWER,
    MANDELBROT_XSET,
    REAL_COEFF_POLY,
    COMPLEX_COEFF_POLY,
    COMPLEX_C_COEFF_POLY
};

class Fractal2DScene : public Scene {
    public:
        Fractal2DScene(const vec2& dimensions = vec2(1,1));
        StateSet makeFractalStateSet(vector<float> params, const string& identifier, const int mode);
        void draw() override;
    private:
        float origin_params[28];
        float X_params[28];
        float Y_params[28];
        float x_params[28];
        float y_params[28];
        void populateParamArray(float*, const string&);
        void populateAllArrays();
};
