#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& l, double scale, const double width = 1, const double height = 1)
    : ConvolutionScene(width, height), latex(l), scale_factor(scale) {}

    void begin_latex_transition(const string& l) {
        latex = l;
        ScalingParams sp(scale_factor);
        begin_transition(latex_to_pix(latex, sp));
    }

    void jump_latex(string latex) {
        ScalingParams sp(scale_factor);
        jump(latex_to_pix(latex, sp));
    }

private:
    string latex;
    double scale_factor;

protected:
    Pixels get_p1() override {
        ScalingParams sp(scale_factor*get_width(), scale_factor*get_height());
        Pixels ret = latex_to_pix(latex, sp);
        return ret;
    }
};
