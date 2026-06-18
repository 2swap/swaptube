#include "LatexScene.h"
#include <iostream>
#include <stdexcept>
#include "../../Core/Smoketest.h"
#include "../../Core/Convolution.h"
#include "../../IO/SVG.h"

extern "C" Interpolation stage_interpolation(
    const uint32_t* h_pix_1, const ivec2 wh_1, const int num_glyphs_1,
    const uint32_t* h_pix_2, const ivec2 wh_2, const int num_glyphs_2);
extern "C" void interpolate(
    const Interpolation& interpolation, const float t,
    uint32_t* d_output_pix, const ivec2 output_wh);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);
extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);

LatexScene::LatexScene(const string& l, double box_scale, const vec2& dimensions)
: Scene(dimensions), box_scale(box_scale) {
    ScalingParams sp(get_width_height() * box_scale);
    last_pixels = next_pixels = latex_to_pix(l, sp);
    scale_factor = sp.scale_factor;
}

void LatexScene::begin_latex_transition(const TransitionType tt, const string& l) {
    cout << "LatexScene: begin_latex_transition called with TransitionType: " << tt << " and latex: " << l << endl;
    if(transitioning) {
        throw runtime_error("LatexScene: Already transitioning. Cannot begin a new transition until the current one finishes.");
    }
    if(scale_factor == 0) {
        throw runtime_error("LatexScene: scale_factor is not set before begin_latex_transition.");
    }
    ScalingParams sp(scale_factor);
    transitioning = true;
    transition_type = tt;
    last_pixels = next_pixels;
    next_pixels = latex_to_pix(l, sp);
    Pixels last_segmented = segment(last_pixels, last_num_glyphs);
    Pixels next_segmented = segment(next_pixels, next_num_glyphs);
    interp = stage_interpolation(
        last_segmented.pixels.data(), last_pixels.wh, last_num_glyphs,
        next_segmented.pixels.data(), next_pixels.wh, next_num_glyphs
    );
    cout << "LatexScene: Transition started with scale factor: " << sp.scale_factor << endl;
}

void LatexScene::draw() {
    if (transitioning) {
        interpolate(
            interp, smoother2(state["microblock_fraction"]),
            gpu_pix->get_ptr(), get_width_height()
        );
    } else {
        vec2 offset = (get_width_height() - last_pixels.wh) / 2.0f;
        uint32_t* frame_ptr = cuda_alloc_pixels_on_device(last_pixels.wh.x * last_pixels.wh.y);
        cuda_copy_pixels_to_device(last_pixels.pixels.data(), last_pixels.wh.x * last_pixels.wh.y, frame_ptr);
        cuda_overlay(gpu_pix->get_ptr(), get_width_height(), frame_ptr, last_pixels.wh, offset, 1.0f, 0.0f);
        cuda_free_pixels_on_device(frame_ptr);
    }
}

void LatexScene::on_end_transition_extra_behavior(const TransitionType tt) {
    if (transitioning && (transition_type == MICRO || tt == MACRO)) {
        last_pixels = next_pixels;
        transitioning = false;
    }
}

void LatexScene::jump_latex(string l) {
    ScalingParams sp(scale_factor);
    next_pixels = last_pixels = latex_to_pix(l, sp);
    transitioning = false;
}

const StateQuery LatexScene::populate_state_query() const {
    return StateQuery{"microblock_fraction"};
}
