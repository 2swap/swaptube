#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& eqn, double suggested_scale, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : ConvolutionScene(init_latex(eqn, width*suggested_scale, height*suggested_scale), width, height) {}

    void begin_latex_transition(string eqn) {
        ScalingParams sp(scale_factor);
        Pixels p = eqn_to_pix(eqn, sp);
        cout << p.w << " " << p.h << " " << scale_factor << endl;
        begin_transition(p);
    }

private:
    double scale_factor;

    Pixels init_latex(const string& eqn, int width, int height) {
        ScalingParams sp = ScalingParams(width, height);
        Pixels ret = eqn_to_pix(eqn, sp);
        scale_factor = sp.scale_factor;

        return ret;
    }
};
