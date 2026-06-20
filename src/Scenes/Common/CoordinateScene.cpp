#include "CoordinateScene.h"
#include <vector>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <list>
#include <utility>
#include <algorithm>
#include "../../Host_Device_Shared/helpers.h"

using std::string;
using std::ostringstream;
using std::unordered_set;
using std::list;
using std::pair;
using std::min;

extern "C" void draw_coordinate_grid(uint32_t* d_pixels, const ivec2& pix_wh, const vec2& lx_ty, const vec2& rx_by);

string float_to_pretty_string(const float value) {
    if(abs(value) < 0.0000001) return "0";

    // Convert float to string with a stream
    ostringstream oss;
    if(value == (int)value)
        oss << fixed;
    oss << value;
    string str = oss.str();

    // Find the position of the decimal point
    size_t decimalPos = str.find('.');

    // If there's no decimal point, just return the string
    if (decimalPos != string::npos) {
        // Remove trailing zeros
        size_t endPos = str.find_last_not_of('0');

        // If the last non-zero character is the decimal point, remove it too
        if (endPos == decimalPos) {
            endPos--;
        }

        // Create a substring up to the correct position
        str = str.substr(0, endPos + 1);
    }

    if(str.size() > 2 && str[0] == '0' && str[1] == '.') str = str.substr(1);
    if(str.size() > 2 && str[0] == '-' && str[1] == '0') str = "-" + str.substr(2);
    return str;
}

string truncate_tick(const float value, const bool append_i) {
    string str = float_to_pretty_string(value);

    if(append_i) {
        if(str == "0") return "0";
        if(str == "1") return "i";
        if(str == "-1") return "-i";
        return str+"i";
    }
    return str;
}

CoordinateScene::CoordinateScene(const vec2& dimensions)
    : Scene(dimensions) {
    manager.set({
        {"left_x"   , "<center_x> .5 <window_width> / -"},
        {"right_x"  , "<center_x> .5 <window_width> / +"},
        {"top_y"    , "<center_y> .5 <window_height> / -"},
        {"bottom_y" , "<center_y> .5 <window_height> / +"},
        {"ticks_opacity", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "0"},
        {"microblock_fraction_passthrough", "{microblock_fraction}"},
        {"window_height", "<zoom> exp .2 *"},
        {"window_width", "<window_height> <w> {VIDEO_WIDTH} * / <h> {VIDEO_HEIGHT} * *"},
    });
}

vec2 CoordinateScene::point_to_pixel(const vec2& p) {
    const vec2 wh(get_width(), get_height());
    const vec2 rx_by(state["right_x"], state["bottom_y"]);
    const vec2 lx_ty(state["left_x"], state["top_y"]);
    return point_to_pixel_in_screen(p, lx_ty, rx_by, wh);
}

vec2 CoordinateScene::pixel_to_point(const vec2& pix) {
    const vec2 wh(get_width(), get_height());
    const vec2 rx_by(state["right_x"], state["bottom_y"]);
    const vec2 lx_ty(state["left_x"], state["top_y"]);
    return pixel_to_point_in_screen(pix, lx_ty, rx_by, wh);
}

void CoordinateScene::draw() {
    draw_coordinate_grid(gpu_pix->get_ptr(), get_width_height(), vec2(state["left_x"], state["top_y"]), vec2(state["right_x"], state["bottom_y"]));
}

/*
void CoordinateScene::draw_one_axis(bool ymode) {
    const float ticks_opacity = state["ticks_opacity"];
    if(ticks_opacity < 0.01) return;
    const int w = get_width();
    const int h = get_height();
    const float gmsz = get_geom_mean_size();

    const float z = state[ymode?"window_height":"window_width"] + 0.00000001;
    const float upper_bound = ymode?state["bottom_y"]:state["right_x"];
    const float lower_bound = ymode?state[   "top_y"]:state[ "left_x"];
    const float log10z = log10(z);
    int order_mag = floor(-log10z);
    const float log_decimal = log10z-floor(log10z);
    bool fiveish = log_decimal >= .5;
    const float interpolator = (log_decimal >= .5 ? -1 : 0) + log_decimal * 2;
    unordered_set<string> done_numbers;
    for(int d_om = 0; d_om < 2; d_om++){
        float increment = pow(10, order_mag) * (fiveish ? 0.5 : 1);
        for(float x_y = floor(lower_bound/increment)*increment; x_y < upper_bound; x_y += increment) {
            string truncated = truncate_tick(x_y, ymode && complex_plane);
            if(done_numbers.find(truncated) != done_numbers.end()) continue;
            done_numbers.insert(truncated);
            float tick_length = (d_om == 1 ? 2 * interpolator : 2) * gmsz / 192.;
            vec2 point = point_to_pixel(vec2(x_y, x_y));
            float coordinate = ymode?point.y:point.x;
            float number_opacity = d_om == 1 ? (interpolator<.5? 0 : interpolator*2-1) : 1;
            number_opacity *= ticks_opacity * (1-square(square(2.5*(.5-coordinate/(ymode?h:w)))));
            if(number_opacity < 0) number_opacity = 0;
            if(ymode) pix.bresenham(0, coordinate, tick_length, coordinate, OPAQUE_WHITE, number_opacity, 1);
            else      pix.bresenham(coordinate, h-1, coordinate, h-1-tick_length, OPAQUE_WHITE, number_opacity, 1);
            if(number_opacity > 0){
                ScalingParams sp(gmsz * vec2(1/9., 1/18.));
                Pixels latex = latex_to_pix(truncated, sp);
                //if(ymode) latex = latex.rotate_90();
                if(ymode) pix.overlay_cpu(latex, ivec2(tick_length * .8      , coordinate - latex.wh.y*1.1       ), number_opacity);
                if(!ymode)pix.overlay_cpu(latex, ivec2(coordinate - latex.wh.x/2, h-1-tick_length * 1.5 - latex.wh.y), number_opacity);
            }
        }
        if(fiveish) order_mag--;
        fiveish = !fiveish;
    }
}
*/

const StateQuery CoordinateScene::populate_state_query() const {
    StateQuery sq = {"left_x", "right_x", "window_height", "window_width", "top_y", "bottom_y", "ticks_opacity"};
    return sq;
}
