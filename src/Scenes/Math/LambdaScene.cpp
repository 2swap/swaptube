#include "LambdaScene.h"
#include <algorithm>
#include "../../DataObjects/LambdaCalculus/LambdaUtils.h"
#include <cmath>
#include <string>

using namespace std;

LambdaScene::LambdaScene(const shared_ptr<const LambdaExpression> lambda, const double width, const double height) : Scene(width, height), le(lambda->clone()) {
    le->set_positions();
    manager.set({{"latex_opacity", "0"}, {"title_opacity", "0"}});
}

const StateQuery LambdaScene::populate_state_query() const {
    return StateQuery{"microblock_fraction", "title_opacity", "latex_opacity"};
}

void LambdaScene::reduce(){
    le->flush_uid_recursive();
    if(le->is_reducible()) {
        set_expression(le->clone()->reduce());
    }
    else last_le = nullptr;
    render_diagrams();
}

void LambdaScene::set_expression(shared_ptr<LambdaExpression> lambda) {
    last_le = le->clone();
    le = lambda->clone();
    last_le_w = le_pix.w;
    last_le_h = le_pix.h;
    le->set_positions();
}

void LambdaScene::render_diagrams(){
    le->set_positions();
    le_pix = le->draw_lambda_diagram(get_scale(le));
}

void LambdaScene::set_title(string t){
    title = t;
}

shared_ptr<LambdaExpression> LambdaScene::get_clone(){
    return le->clone();
}

// TODO transitions always assumed to be per-microblock, perhaps add macroblock transitioning?
void LambdaScene::on_end_transition_extra_behavior(const TransitionType tt) { last_le = nullptr; }
void LambdaScene::mark_data_unchanged() { le->mark_unchanged(); }
void LambdaScene::change_data() { } // This scene is expected to only be manipulated by the customer
bool LambdaScene::check_if_data_changed() const {
    return le->has_been_updated_since_last_scene_query();
}

float LambdaScene::get_scale(shared_ptr<const LambdaExpression> expr) {
    return min(get_width()/(expr->get_width_recursive() + 4), get_height()/(expr->get_height_recursive() + 4));
}

void LambdaScene::draw() {
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
        Pixels latex = latex_to_pix("\\text{" + title + "}", sp);
        pix.overlay(latex, (pix.w-latex.w)*.5, pix.h*7/8-latex.h, state["title_opacity"]);
    }
}
