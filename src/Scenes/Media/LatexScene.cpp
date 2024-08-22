#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& eqn, double extra_scale, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : first_scaling_params(pix.w * extra_scale, pix.h * extra_scale), ConvolutionScene(eqn_to_pix(eqn, first_scaling_params), width, height), equation_string(eqn) {
        scale_factor = first_scaling_params.scale_factor;
        TODO the initializer list is not in the right order
    }

    void append_latex_transition(string eqn) {
        if(in_transition_state) end_transition();
        begin_latex_transition(equation_string + eqn);
    }

    void begin_latex_transition(string eqn) {
        cout << "rendering latex: " << eqn << endl;
        ScalingParams sp(scale_factor);
        transition_equation_pixels = eqn_to_pix(transition_equation_string, sp);
        cout << equation_string << " <- Finding Intersections -> " << eqn << endl;
    }

private:
    ScalingParams first_scaling_params;
    double scale_factor;
};
