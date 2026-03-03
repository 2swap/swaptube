#pragma once

#include <string>
#include "../Common/ConvolutionScene.h"

class SvgScene : public ConvolutionScene {
public:
    SvgScene(const std::string& l, double box_scale, const vec2& dimensions = vec2(1, 1));

    void begin_svg_transition(const TransitionType tt, const std::string& l);
    void jump_svg(std::string l);

private:
    std::string latex;
    double scale_factor = 0;
    double box_scale;

protected:
    Pixels get_p1() override;
};
