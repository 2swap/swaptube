#include "LambdaScene.h"
#include <algorithm>
#include <cmath>
#include <string>

using namespace std;

extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_overlay(
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle);

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
    // TODO I am lazy and have not refactored this to be on the GPU. This is really dumb:
    // it draws on the CPU, copies to GPU, and overwrites the pixel buffer on GPU.
    const vec2 wh(get_width_height());
    if(last_le == nullptr){
        render_diagrams(); 
        uint32_t* d_pix = cuda_alloc_pixels_on_device(le_pix.wh.x * le_pix.wh.y);
        cuda_copy_pixels_to_device(le_pix.pixels.data(), le_pix.wh.x * le_pix.wh.y, d_pix);
        const ivec2 draw_pos = floor((wh - vec2(le_pix.wh)) * .5);
        cuda_overlay(gpu_pix->get_ptr(), get_width_height(), d_pix, le_pix.wh, draw_pos, 1, 0);
        cuda_free_pixels_on_device(d_pix);
    } else {
        float trans_frac = state["microblock_fraction"];
        pair<shared_ptr<LambdaExpression>, shared_ptr<LambdaExpression>> interpolated = get_interpolated(last_le, le, trans_frac);
        float scale = smoothlerp(get_scale(last_le), get_scale(le), trans_frac);
        Pixels p1 = interpolated.first->draw_lambda_diagram(scale);
        Pixels p2 = interpolated.second->draw_lambda_diagram(scale);

        // Copy p1 and p2 to gpu
        uint32_t* d_p1 = cuda_alloc_pixels_on_device(p1.wh.x * p1.wh.y);
        uint32_t* d_p2 = cuda_alloc_pixels_on_device(p2.wh.x * p2.wh.y);
        cuda_copy_pixels_to_device(p1.pixels.data(), p1.wh.x * p1.wh.y, d_p1);
        cuda_copy_pixels_to_device(p2.pixels.data(), p2.wh.x * p2.wh.y, d_p2);

        vec2 pixwh(smoothlerp(p1.wh.x, p2.wh.x, trans_frac), smoothlerp(p1.wh.y, p2.wh.y, trans_frac));
        const ivec2 draw_pos = floor((wh - pixwh) * .5);
        cuda_overlay(gpu_pix->get_ptr(), get_width_height(), d_p1, p1.wh, draw_pos, 1, 0);
        cuda_overlay(gpu_pix->get_ptr(), get_width_height(), d_p2, p2.wh, draw_pos, 1, 0);

        cuda_free_pixels_on_device(d_p1);
        cuda_free_pixels_on_device(d_p2);
    }
}
