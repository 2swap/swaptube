#include "StateSliderScene.h"
#include <sstream>
#include <iomanip>
#include <cmath>

using std::string;
using std::ostringstream;
using std::ostringstream;
using std::fixed;
using std::setprecision;

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);
extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);
extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);

string double_to_string(double value) {
    const int sig_figs = 2;
    ostringstream out;
    if (value == 0) { return "0"; }

    int exponent = static_cast<int>(std::floor(std::log10(std::abs(value))));
    int significant_digits = sig_figs - exponent - 1;

    out << std::fixed << std::setprecision(significant_digits) << value;
    string str = out.str();
    
    // Remove trailing zeros
    str.erase(str.find_last_not_of('0') + 1, string::npos);
    
    // Remove decimal point if it is the last character
    if (!str.empty() && str.back() == '.') {
        str.pop_back();
    }
    
    return str;
}

StateSliderScene::StateSliderScene(const string& vn, const string& dn, double min_val, double max_val, const vec2& dimensions)
: Scene(dimensions), display_name(dn), min_value(min_val), max_value(max_val) {
    manager.set("value", vn);
}

void StateSliderScene::draw() {
    const ivec2 wh(get_width_height());
    ScalingParams sp(wh * vec2(1, .6));
    draw_slider();
    if(display_name != "") {
        string eqn_str = display_name + " = " + double_to_string(state["value"]);
        Pixels equation_pixels = latex_to_pix(eqn_str, sp);
        vec2 text_pos(0, (wh.y-equation_pixels.wh.y)/2.);
        uint32_t* d_equation_pixels = cuda_alloc_pixels_on_device(equation_pixels.wh.x * equation_pixels.wh.y);
        cuda_copy_pixels_to_device(equation_pixels.pixels.data(), equation_pixels.wh.x * equation_pixels.wh.y, d_equation_pixels);
        cuda_overlay(gpu_pix->get_ptr(), wh, d_equation_pixels, equation_pixels.wh, text_pos, 1.0, 0.0);
        cuda_free_pixels_on_device(d_equation_pixels);
    }
}

const StateQuery StateSliderScene::populate_state_query() const {
    return StateQuery{"value"};
}

void StateSliderScene::draw_slider() {
    const ivec2 wh(get_width_height());

    const int knob_color = 0xff444444; // Dark gray

    double value = state["value"];
    double normalized_value = (value - min_value) / (max_value - min_value);

    uint32_t bar_color = 0xff888888;

    ivec2 tl(wh.y * .5, wh.y * .4 + 1);
    ivec2 br(wh.x - wh.y * .5, wh.y * .6 + 1);
    draw_rectangle(gpu_pix->get_ptr(), wh, tl, br, bar_color);

    draw_circle(gpu_pix->get_ptr(), wh, ivec2(wh.y * .5, wh.y * .5), wh.y * .1, bar_color);
    draw_circle(gpu_pix->get_ptr(), wh, ivec2(wh.x-wh.y*.5,wh.y*.5), wh.y * .1, bar_color);

    vec2 center(wh.y * .5 + normalized_value * (wh.x - wh.y), wh.y/2.);
    draw_circle(gpu_pix->get_ptr(), wh, center, wh.y * .5, knob_color);
}
