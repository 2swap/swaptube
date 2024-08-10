#pragma once

#include "../../DataObjects/LambdaCalculus.cpp"
#include "../../io/VisualMedia.cpp"

class LambdaScene : public Scene {
public:
    LambdaScene(const string& lambda_str                     , const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height), le(parse_lambda_from_string(lambda_str)) { }
    LambdaScene(const shared_ptr<LambdaExpression> lambda_str, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height), le(lambda_str) { }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

    void reduce(){
        if(le->is_reducible())
            le = le->reduce();
    }

    void mark_data_unchanged() override { le->mark_unchanged(); }
    void change_data() override { } // This scene is expected to only be manipulated by the customer
    bool check_if_data_changed() const override {
        return le->has_been_updated_since_last_scene_query();
    }

    void draw() override {
        pix.fill(TRANSPARENT_BLACK);
        Pixels lambda_pixels(le->draw_lambda_diagram(), 8);
        ScalingParams sp(pix.w, pix.h / 2);
        Pixels latex = eqn_to_pix(le->get_latex(), sp);
        pix.overwrite(lambda_pixels, (pix.w-lambda_pixels.w)*.5, (pix.h-lambda_pixels.h)*.5-pix.h*.25);
        pix.overwrite(latex        , (pix.w-        latex.w)*.5, (pix.h-        latex.h)*.5+pix.h*.25);
    }

private:
    // Things used for non-transition states
    shared_ptr<LambdaExpression> le;
    int tick = 0;
};
