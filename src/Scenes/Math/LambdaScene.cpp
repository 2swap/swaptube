#pragma once

#include "../../DataObjects/LambdaCalculus.cpp"
#include "../../io/VisualMedia.cpp"

class LambdaScene : public Scene {
public:
    LambdaScene(const shared_ptr<const LambdaExpression> lambda, const double width = 1, const double height = 1) : Scene(width, height), le(lambda->clone()) {
        le->set_positions();
        state_manager.set({{"latex_opacity", "0"},
                           {"title_opacity", "0"}});
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"microblock_fraction", "title_opacity", "latex_opacity"};
    }

    void reduce(){
        le->flush_uid_recursive();
        if(le->is_reducible()) {
            set_expression(le->clone()->reduce());
        }
        else last_le = nullptr;
        render_diagrams();
    }

    void set_expression(shared_ptr<LambdaExpression> lambda) {
        last_le = le->clone();
        le = lambda->clone();
        last_le_w = le_pix.w;
        last_le_h = le_pix.h;
        le->set_positions();
    }

    void render_diagrams(){
        le->set_positions();
        le_pix = le->draw_lambda_diagram(get_scale(le));
    }

    void set_title(string t){
        title = t;
    }

    void on_end_transition_extra_behavior(bool is_macroblock) override {last_le = nullptr;}
    void mark_data_unchanged() override { le->mark_unchanged(); }
    void change_data() override { } // This scene is expected to only be manipulated by the customer
    bool check_if_data_changed() const override {
        return le->has_been_updated_since_last_scene_query();
    }

    float get_scale(shared_ptr<const LambdaExpression> expr) {
        return min(get_width()/(expr->get_width_recursive() + 4), get_height()/(expr->get_height_recursive() + 4));
    }

    void draw() override {
        if(last_le == nullptr){
            render_diagrams(); 
            pix.overwrite(le_pix, (pix.w-le_pix.w)*.5, (pix.h-le_pix.h)*.5);
        } else {
            float trans_frac = state["microblock_fraction"];
            pair<shared_ptr<LambdaExpression>, shared_ptr<LambdaExpression>> interpolated = get_interpolated(last_le, le, trans_frac);
            float scale = smoothlerp(get_scale(last_le), get_scale(le), trans_frac);
            Pixels p1 = interpolated.first->draw_lambda_diagram(scale);
            Pixels p2 = interpolated.second->draw_lambda_diagram(scale);
            float pixw = smoothlerp(p1.w, p2.w, trans_frac);
            float pixh = smoothlerp(p1.h, p2.h, trans_frac);
            pix.overwrite(p1, (pix.w-pixw)*.5, (pix.h-pixh)*.5);
            pix.overlay  (p2, (pix.w-pixw)*.5, (pix.h-pixh)*.5);
        }
        if(state["latex_opacity"] > 0.01){
            ScalingParams sp(pix.w, pix.h / 4);
            Pixels latex = latex_to_pix(le->get_latex(), sp);
            pix.overlay(latex, (pix.w-latex.w)*.5, pix.h*7/8-latex.h, state["latex_opacity"]);
        }
        if(state["title_opacity"] > 0.01){
            ScalingParams sp(pix.w, pix.h / 4);
            Pixels latex = latex_to_pix(latex_text(title), sp);
            pix.overlay(latex, (pix.w-latex.w)*.5, pix.h*7/8-latex.h, state["title_opacity"]);
        }
    }

private:
    shared_ptr<LambdaExpression> le;
    shared_ptr<LambdaExpression> last_le;
    Pixels le_pix;
    int last_le_w = 0;
    int last_le_h = 0;
    int tick = 0;
    string title;
};
