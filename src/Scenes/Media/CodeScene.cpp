#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class CodeScene: public ConvolutionScene {
public:
    CodeScene(const string& code, double suggested_scale, const double width = 1, const double height = 1)
    : ConvolutionScene(init_code(code, width*suggested_scale, height*suggested_scale), width, height) {}

    void begin_code_transition(string code) {
        ScalingParams sp(scale_factor);
        Pixels p = code_to_pix(code, sp);
        cout << p.w << " " << p.h << " " << scale_factor << endl;
        begin_transition(p);
    }

private:
    double scale_factor;

    Pixels init_code(const string& code, int width, int height) {
        ScalingParams sp = ScalingParams(width, height);
        Pixels ret = code_to_pix(code, sp);
        scale_factor = sp.scale_factor;

        return ret;
    }
};
