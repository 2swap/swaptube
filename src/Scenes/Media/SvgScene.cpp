#include "SvgScene.h"

#include <librsvg-2.0/librsvg/rsvg.h>
#include <stdexcept>

#include "../../IO/VisualMedia.h"
#include "../../Core/Smoketest.h"

using std::string;

SvgScene::SvgScene(const string& l, double box_scale, const double width, const double height)
: ConvolutionScene(width, height), latex(l), box_scale(box_scale) {}

void SvgScene::begin_svg_transition(const TransitionType tt, const string& l) {
    latex = l;
    if(scale_factor == 0) {
        if(!rendering_on()) {
            scale_factor = 1;
        } else {
            throw std::runtime_error("LatexScene: scale_factor is not set before begin_latex_transition.");
        }
    }
    ScalingParams sp(scale_factor);
    begin_transition(tt, svg_to_pix(latex, sp));
}

void SvgScene::jump_svg(string l) {
    // If you wanted to re-initialize the scale:
    //ScalingParams sp(scale_factor*get_width(), scale_factor*get_height());

    //But we usually don't do that
    latex = l;
    ScalingParams sp(scale_factor);
    jump(svg_to_pix(l, sp));
}

Pixels SvgScene::get_p1() {
    ScalingParams sp = scale_factor == 0 ? ScalingParams(box_scale*get_width(), box_scale*get_height()) : ScalingParams(scale_factor);
    Pixels ret = svg_to_pix(latex, sp);
    scale_factor = sp.scale_factor;
    return ret;
}
