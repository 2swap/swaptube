#include "LambdaScene.h"
#include <algorithm>
#include <cmath>
#include <string>

using namespace std;

LambdaScene::LambdaScene(const shared_ptr<const LambdaExpression> lambda, const vec2& dimensions) :
    Scene(dimensions), le(lambda->clone()) {
    le->set_positions();
}

const StateQuery LambdaScene::populate_state_query() const {
    return StateQuery{"microblock_fraction"};
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
    last_le_w = le_pix.wh.x;
    last_le_h = le_pix.wh.y;
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

float LambdaScene::get_scale(shared_ptr<const LambdaExpression> expr) {
    return min(get_width()/(expr->get_width_recursive() + 4), get_height()/(expr->get_height_recursive() + 4));
}

void LambdaScene::draw() {
    if(last_le == nullptr){
        render_diagrams(); 
        pix.overwrite(le_pix, (pix.wh-le_pix.wh)*.5);
    } else {
        float trans_frac = state["microblock_fraction"];
        pair<shared_ptr<LambdaExpression>, shared_ptr<LambdaExpression>> interpolated = get_interpolated(last_le, le, trans_frac);
        float scale = smoothlerp(get_scale(last_le), get_scale(le), trans_frac);
        Pixels p1 = interpolated.first->draw_lambda_diagram(scale);
        Pixels p2 = interpolated.second->draw_lambda_diagram(scale);
        vec2 pixwh(smoothlerp(p1.wh.x, p2.wh.x, trans_frac), smoothlerp(p1.wh.y, p2.wh.y, trans_frac));
        pix.overwrite  (p1, (pix.wh-pixwh)*.5);
        pix.overlay_gpu(p2, (pix.wh-pixwh)*.5);
    }
}
