#pragma once

#include "LambdaExpression.cpp"
#include "../../../io/VisualMedia.cpp"

class LambdaScene : public Scene {
public:
    LambdaScene(const string& lambda_str, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : Scene(width, height), le(parse_lambda_from_string(lambda_str)) { }

    void draw() override {
        pix.fill(TRANSPARENT_BLACK);
        if(le->is_reducible() && (++tick%5==4)) le = le->reduce();
        Pixels lambda_pixels(le->draw_lambda_diagram(), 5);
        ScalingParams sp(pix.w, pix.h / 2);
        Pixels latex = eqn_to_pix(le->get_latex(), sp);
        pix.overwrite(lambda_pixels, (pix.w-lambda_pixels.w)*.5, (pix.h-lambda_pixels.h)*.5-pix.h*.25);
        pix.overwrite(latex        , (pix.w-        latex.w)*.5, (pix.h-        latex.h)*.5+pix.h*.25);
    }

    bool scene_requests_rerender() const override { return true; }

private:
    // Things used for non-transition states
    shared_ptr<LambdaExpression> le;
    int tick = 0;
};
