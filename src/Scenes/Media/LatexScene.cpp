#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& l, double scale, const double width = 1, const double height = 1)
    : ConvolutionScene(width, height), latex(l), scale_factor(scale) {}

    void begin_latex_transition(const TransitionType tt, const string& l) {
        cout << "LatexScene: begin_latex_transition called with TransitionType: " << tt << " and latex: " << l << endl;
        latex = l;
        ScalingParams sp(scale);
        cout << "B" << endl;
        begin_transition(tt, latex_to_pix(latex, sp));
        cout << "LatexScene: Transition started with scale factor: " << sp.scale_factor << endl;
    }

    void jump_latex(string latex) {
        // If you wanted to re-initialize the scale:
        //ScalingParams sp(scale_factor*get_width(), scale_factor*get_height());

        //But we usually don't do that
        ScalingParams sp(scale);
        jump(latex_to_pix(latex, sp));
    }

private:
    string latex;
    double scale = 0;
    double scale_factor;

protected:
    Pixels get_p1() override {
        ScalingParams sp = scale == 0 ? ScalingParams(scale_factor*get_width(), scale_factor*get_height()) : ScalingParams(scale);
        Pixels ret = latex_to_pix(latex, sp);
        scale = sp.scale_factor;
        return ret;
    }
};
