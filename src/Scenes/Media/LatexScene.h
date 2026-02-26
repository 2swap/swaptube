#pragma once

#include "../Common/ConvolutionScene.h"

class LatexScene : public ConvolutionScene {
public:
    LatexScene(const string& l, double box_scale, const double width = 1, const double height = 1);

    void begin_latex_transition(const TransitionType tt, const string& l);

    void jump_latex(string l);

private:
    string latex;
    double scale_factor = 0;
    double box_scale;

protected:
    Pixels get_p1() override;
};
