#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& latex, double suggested_scale, const double width = 1, const double height = 1)
    : ConvolutionScene(init_latex(latex, width*suggested_scale, height*suggested_scale), width, height) {}

    void begin_latex_transition(string latex) {
        ScalingParams sp(scale_factor);
        Pixels p = latex_to_pix(latex, sp);
        begin_transition(p);
    }

private:
    double scale_factor;

    Pixels init_latex(const string& latex, int width, int height) {
        ScalingParams sp = ScalingParams(width, height);
        Pixels ret = latex_to_pix(latex, sp);
        scale_factor = sp.scale_factor;

        return ret;
    }
};
