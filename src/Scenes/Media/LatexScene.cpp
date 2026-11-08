#include "LatexScene.h"
#include <iostream>
#include <stdexcept>
#include "../../Core/Smoketest.h"
#include "../../Core/GlyphSegmentation.h"
#include "../../IO/Latex.h"

extern "C" Interpolation stage_interpolation(
    const uint32_t* h_pix_1, const uint32_t* h_vis_1, const uint32_t* h_shapes_1, const ivec2 wh_1, const int num_glyphs_1,
    const uint32_t* h_pix_2, const uint32_t* h_vis_2, const uint32_t* h_shapes_2, const ivec2 wh_2, const int num_glyphs_2);
extern "C" void interpolate(
    const Interpolation& interpolation, const float t,
    uint32_t* d_output_pix, const ivec2 output_wh);
extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);
extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);

// Shape class of each glyph (index label - 1), from the SVG element each glyph came from.
static vector<uint32_t> shapes_of_glyphs(const vector<uint32_t>& element_of_label, const vector<uint32_t>& shape_of_element) {
    vector<uint32_t> shapes;
    for (size_t label = 1; label < element_of_label.size(); label++) {
        const uint32_t element = element_of_label[label];
        shapes.push_back(element < shape_of_element.size() ? shape_of_element[element] : 0);
    }
    return shapes;
}

LatexScene::LatexScene(const string& l, const vec2& dimensions)
: Scene(dimensions) {
    ScalingParams sp(get_width_height());
    current_scaling = sp;
    last_pixels = next_pixels = latex_to_gpu_pix(l, sp);
    current_latex = l;
}

void LatexScene::begin_latex_transition(const TransitionType tt, const string& l) {
    cout << "LatexScene: begin_latex_transition called with TransitionType: " << tt << " and latex: " << l << endl;
    if(transitioning) {
        throw runtime_error("LatexScene: Already transitioning. Cannot begin a new transition until the current one finishes.");
    }
    ScalingParams sp(get_width_height());
    const ScalingParams fresh_sp = sp; // latex_to_gpu_pix mutates its argument
    transitioning = true;
    transition_type = tt;
    last_pixels = next_pixels;
    next_pixels = latex_to_gpu_pix(l, sp);
    Pixels last_pixels_cpu(last_pixels->get_wh());
    Pixels next_pixels_cpu(next_pixels->get_wh());
    ivec2 last_wh = last_pixels_cpu.wh;
    ivec2 next_wh = next_pixels_cpu.wh;
    cuda_copy_pixels_to_host(last_pixels_cpu.pixels.data(), last_wh.x*last_wh.y, last_pixels->get_ptr());
    cuda_copy_pixels_to_host(next_pixels_cpu.pixels.data(), next_wh.x*next_wh.y, next_pixels->get_ptr());
    vector<uint32_t> last_elements, next_elements;
    Pixels last_segmented = segment_by_glyph_ids(last_pixels_cpu, latex_to_glyph_ids(current_latex, current_scaling), last_num_glyphs, last_elements);
    Pixels next_segmented = segment_by_glyph_ids(next_pixels_cpu, latex_to_glyph_ids(l, fresh_sp), next_num_glyphs, next_elements);
    const vector<uint32_t> last_shapes = shapes_of_glyphs(last_elements, latex_glyph_shapes(current_latex));
    const vector<uint32_t> next_shapes = shapes_of_glyphs(next_elements, latex_glyph_shapes(l));
    current_latex = l;
    current_scaling = fresh_sp;
    interp = stage_interpolation(
        last_segmented.pixels.data(), last_pixels_cpu.pixels.data(), last_shapes.data(), last_pixels_cpu.wh, last_num_glyphs,
        next_segmented.pixels.data(), next_pixels_cpu.pixels.data(), next_shapes.data(), next_pixels_cpu.wh, next_num_glyphs
    );
}

void LatexScene::draw() {
    if (transitioning) {
        float fraction = state[transition_type == MACRO ? "macroblock_fraction" : "microblock_fraction"];
        interpolate(
            interp, smoother2(fraction),
            gpu_pix.get_ptr(), get_width_height()
        );
    } else {
        vec2 offset = get_width_height() / 2.0f;
        cuda_overlay(gpu_pix.get_ptr(), get_width_height(), last_pixels->get_ptr(), last_pixels->get_wh(), offset, 1.0f, 0.0f);
    }
}

void LatexScene::on_end_transition_extra_behavior(const TransitionType tt) {
    if (transitioning && (transition_type == MICRO || tt == MACRO)) {
        last_pixels = next_pixels;
        transitioning = false;
    }
}

void LatexScene::jump_latex(string l) {
    ScalingParams sp(get_width_height());
    current_scaling = sp;
    next_pixels = last_pixels = latex_to_gpu_pix(l, sp);
    current_latex = l;
    transitioning = false;
}
