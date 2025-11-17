#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../Common/ConvolutionScene.cpp"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& l, double box_scale, const double width = 1, const double height = 1)
    : ConvolutionScene(width, height), latex(l), box_scale(box_scale) {}

    void begin_latex_transition(const TransitionType tt, const string& l) {
        cout << "LatexScene: begin_latex_transition called with TransitionType: " << tt << " and latex: " << l << endl;
        latex = l;
        if(scale_factor == 0) {
            if(!rendering_on()) {
                cout << "LatexScene: Warning - scale_factor is not set before begin_latex_transition. Defaulting to 1." << endl;
                scale_factor = 1;
            } else {
                throw runtime_error("LatexScene: scale_factor is not set before begin_latex_transition.");
            }
        }
        ScalingParams sp(scale_factor);
        begin_transition(tt, latex_to_pix(latex, sp));
        cout << "LatexScene: Transition started with scale factor: " << sp.scale_factor << endl;
    }

    void jump_latex(string l) {
        // If you wanted to re-initialize the scale:
        //ScalingParams sp(scale_factor*get_width(), scale_factor*get_height());

        //But we usually don't do that
        latex = l;
        ScalingParams sp(scale_factor);
        jump(latex_to_pix(l, sp));
    }

private:
    string latex;
    double scale_factor = 0;
    double box_scale;

protected:
    Pixels get_p1() override {
        ScalingParams sp = scale_factor == 0 ? ScalingParams(box_scale*get_width(), box_scale*get_height()) : ScalingParams(scale_factor);
        Pixels ret = latex_to_pix(latex, sp);
        scale_factor = sp.scale_factor;
        return ret;
    }
};
