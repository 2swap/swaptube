#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& latex, double suggested_scale, const double width = 1, const double height = 1)
    : ConvolutionScene(init_latex(latex, width*suggested_scale*VIDEO_WIDTH, height*suggested_scale*VIDEO_HEIGHT), width, height), sp(ScalingParams(width, height)) {}

    void begin_latex_transition(string latex) {
        begin_transition(latex_to_pix(latex, sp));
    }

    void jump_latex(string latex) {
        jump(latex_to_pix(latex, sp));
    }

private:
    ScalingParams sp;

    Pixels init_latex(const string& latex, int width, int height) {
        return latex_to_pix(latex, sp);
    }
};
