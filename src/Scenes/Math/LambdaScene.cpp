#pragma once

#include "../../DataObjects/LambdaCalculus.cpp"
#include "../../io/VisualMedia.cpp"

class LambdaScene : public Scene {
public:
    LambdaScene(const string& lambda_str                     , const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height), le(parse_lambda_from_string(lambda_str)) { }
    LambdaScene(const shared_ptr<LambdaExpression> lambda_str, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height), le(lambda_str) { le->set_positions(); le_pix = le->draw_lambda_diagram(8); }

    const StateQuery populate_state_query() const override {
        return StateQuery{"subscene_transition_fraction"};
    }

    void reduce(){
        last_le = le->clone();
        last_le->set_positions();
        last_le_pix = last_le->draw_lambda_diagram(8);
        if(le->is_reducible()){
            le = le->reduce();
            le->set_positions();
            le_pix = le->draw_lambda_diagram(8);
        }
    }

    void mark_data_unchanged() override { le->mark_unchanged(); }
    void change_data() override { } // This scene is expected to only be manipulated by the customer
    bool check_if_data_changed() const override {
        return le->has_been_updated_since_last_scene_query();
    }

    void draw() override {
        pix.fill(TRANSPARENT_BLACK);
        Pixels lambda_pixels;
        float pixw; float pixh;
        if(last_le == nullptr){
            pixw = le_pix.w;
            pixh = le_pix.h;
            lambda_pixels = le_pix;
        } else {
            shared_ptr<LambdaExpression> interpolated = get_interpolated(last_le, le, state["subscene_transition_fraction"]);
            pixw = smoothlerp(last_le_pix.w, le_pix.w, state["subscene_transition_fraction"]);
            pixh = smoothlerp(last_le_pix.h, le_pix.h, state["subscene_transition_fraction"]);
            lambda_pixels = interpolated->draw_lambda_diagram(8);
        }
        ScalingParams sp(pix.w, pix.h / 2);
        Pixels latex = eqn_to_pix(le->get_latex(), sp);
        pix.overwrite(lambda_pixels, (pix.w-pixw)   *.5, (pix.h-pixh)   *.5 - pix.h*.25);
        pix.overwrite(latex        , (pix.w-latex.w)*.5, (pix.h-latex.h)*.5 + pix.h*.25);
    }

private:
    // Things used for non-transition states
    shared_ptr<LambdaExpression> le;
    shared_ptr<LambdaExpression> last_le;
    Pixels le_pix;
    Pixels last_le_pix;
    int tick = 0;
};
