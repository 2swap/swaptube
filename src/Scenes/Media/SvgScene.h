#pragma once

#include <string>
#include "../Common/ConvolutionScene.h"

class SvgScene : public ConvolutionScene {
public:
    SvgScene(const std::string& l, double box_scale, const double width = 1, const double height = 1);

    void begin_svg_transition(const TransitionType tt, const std::string& l);
    void jump_svg(std::string l);

private:
    std::string latex;
    double scale_factor = 0;
    double box_scale;

protected:
    Pixels get_p1() override;
};
