#include "CoordinateScene.h"
#include <vector>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <list>
#include <utility>
#include <algorithm>

using std::string;
using std::ostringstream;
using std::unordered_set;
using std::list;
using std::pair;
using std::min;

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

CoordinateScene::CoordinateScene(const float width, const float height)
    : Scene(width, height) {
    manager.set({
        {"left_x"   , "<center_x> .5 <window_width> / -"},
        {"right_x"  , "<center_x> .5 <window_width> / +"},
        {"top_y"    , "<center_y> .5 <window_height> / -"},
        {"bottom_y" , "<center_y> .5 <window_height> / +"},
        {"construction_opacity", "1"},
        {"ticks_opacity", "0"},
        {"zero_crosshair_opacity", "0"},
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
    const vec2 flip(wh * (p-lx_ty)/(rx_by-lx_ty));
    return vec2(flip.x, get_height()-1-flip.y);
}

vec2 CoordinateScene::pixel_to_point(const vec2& pix) {
    const vec2 wh(get_width(), get_height());
    const vec2 rx_by(state["right_x"], state["bottom_y"]);
    const vec2 lx_ty(state["left_x"], state["top_y"]);
    const vec2 flip(pix.x, get_height()-1-pix.y);
    return lx_ty + (rx_by-lx_ty) * flip / wh;
}

// This is not used here, but it is used in some classes which inherit from CoordinateScene
void CoordinateScene::draw_trail(const list<pair<vec2, int>>& trail, const float trail_opacity) {
    if(trail.size() == 0) return;
    if(trail_opacity < 0.01) return;
    float line_width = get_geom_mean_size()/500.;
    vec2 last_pixel{0,0};
    int i = 0;
    for(const pair<vec2, int>& p : trail) {
        if(i != 0) {
            const vec2 next_pixel(point_to_pixel(p.first));
            pix.bresenham(last_pixel.x, last_pixel.y, next_pixel.x, next_pixel.y, p.second, trail_opacity, line_width);
        }
        last_pixel = point_to_pixel(p.first);
        i++;
    }
}

void CoordinateScene::draw_point(const vec2 point, int point_color, float point_opacity) {
    const vec2 pixel = point_to_pixel(point);
    pix.fill_circle(pixel.x, pixel.y, get_geom_mean_size()/100., point_color, point_opacity);
}

void CoordinateScene::draw() {
    draw_axes();
    draw_zero_crosshair();
    draw_construction();
}

void CoordinateScene::draw_construction() {
    if(construction.size() == 0) return;
    const float construction_opacity = state["construction_opacity"];
    if(construction_opacity < 0.01) return;

    double gm = get_geom_mean_size();
    double line_thickness = gm/200.;
    int point_color = 0xffffffff;
    int line_color = 0xff6666ff;
    int text_color = 0xffffffff;
    float microblock_fraction = 0.5;
    if(state.contains("microblock_fraction_passthrough")) microblock_fraction = state["microblock_fraction_passthrough"];

    float bounce = 1 - square(square(microblock_fraction - 1));
    float interp = smoother2(microblock_fraction);

    Pixels geometry(pix.w, pix.h);

    for(const GeometricLine& l : construction.lines) {
        if(!l.draw_shape) continue;
        vec2 start_point = l.start;
        vec2 end_point = l.end;
        if(l.use_state) {
            start_point = vec2(state["line_"+l.identifier+"_start_x"], state["line_"+l.identifier+"_start_y"]);
            end_point = vec2(state["line_"+l.identifier+"_end_x"], state["line_"+l.identifier+"_end_y"]);
        }
        vec2 start_pixel = point_to_pixel(start_point);
        vec2 end_pixel = point_to_pixel(end_point);
        const vec2 mid_pixel = (start_pixel + end_pixel) / 2.f;
        if(!l.old) {
            // Multiply line length by bounce
            start_pixel = mid_pixel + (start_pixel - mid_pixel) * bounce;
            end_pixel = mid_pixel + (end_pixel - mid_pixel) * bounce;
        }
        geometry.bresenham(start_pixel.x, start_pixel.y, end_pixel.x, end_pixel.y, line_color, 1, line_thickness*.75);
    }
    for(const GeometricPoint& p : construction.points) {
        vec2 position = p.position;
        if(p.use_state) position = vec2(state["point_"+p.identifier+"_x"], state["point_"+p.identifier+"_y"]);
        const vec2 position_pixel = point_to_pixel(position);
        double radius = line_thickness * p.width_multiplier * 2;
        if(p.draw_shape){
            if(!p.old) {
                double radius_pop = line_thickness * p.width_multiplier * 8 * bounce;
                radius = min(radius, radius_pop);
                geometry.fill_circle(position_pixel.x, position_pixel.y, radius_pop, point_color, (1-interp)*.8);
            }
            geometry.fill_circle(position_pixel.x, position_pixel.y, radius, point_color, 1);
        }
        if(p.label != "" && p.width_multiplier > .4) {
            ScalingParams sp(line_thickness * 160 * p.width_multiplier, line_thickness * 16 * p.width_multiplier);
            Pixels latex = latex_to_pix(latex_color(text_color, p.label), sp);
            geometry.overlay(latex, position_pixel.x - latex.w/2, position_pixel.y - line_thickness * 6 - latex.h/2, p.old ? 1 : interp);
        }
    }

    pix.overlay(geometry, 0, 0, construction_opacity);
}

void CoordinateScene::draw_zero_crosshair() {
    const float zc_opacity = state["zero_crosshair_opacity"];
    if(zc_opacity < 0.01) return;
    const int w = get_width();
    const int h = get_height();
    const float gmsz = get_geom_mean_size();
    const vec2 zero = point_to_pixel(vec2(0,0));
    pix.bresenham(zero.x, 0, zero.x, h-1.f, OPAQUE_WHITE, zc_opacity, gmsz/400.);
    pix.bresenham(0, zero.y, w-1.f, zero.y, OPAQUE_WHITE, zc_opacity, gmsz/400.);
}

void CoordinateScene::draw_axes() {
    draw_one_axis(true);
    draw_one_axis(false);
}

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
                ScalingParams sp(gmsz/9., gmsz/18.);
                Pixels latex = latex_to_pix(truncated, sp);
                //if(ymode) latex = latex.rotate_90();
                if(ymode) pix.overlay(latex, tick_length * .8/*1.5*/, coordinate - latex.h*1.1/*.5*/, number_opacity);
                if(!ymode)pix.overlay(latex, coordinate - latex.w/2, h-1-tick_length * 1.5 - latex.h, number_opacity);
            }
        }
        if(fiveish) order_mag--;
        fiveish = !fiveish;
    }
}

const StateQuery CoordinateScene::populate_state_query() const {
    StateQuery sq = {"left_x", "right_x", "window_height", "window_width", "top_y", "bottom_y", "ticks_opacity", "construction_opacity", "zero_crosshair_opacity"};
    for(const GeometricPoint& p : construction.points) {
        if(!p.old) {
            sq.insert("microblock_fraction_passthrough");
            break;
        }
    }
    for(const GeometricLine& l : construction.lines) {
        if(!l.old) {
            sq.insert("microblock_fraction_passthrough");
            break;
        }
    }
    for (const GeometricPoint& p : construction.points) {
        if(p.use_state) {
            sq.insert("point_"+p.identifier+"_x");
            sq.insert("point_"+p.identifier+"_y");
        }
    }
    for (const GeometricLine& l : construction.lines) {
        if(l.use_state) {
            sq.insert("line_"+l.identifier+"_start_x");
            sq.insert("line_"+l.identifier+"_start_y");
            sq.insert("line_"+l.identifier+"_end_x");
            sq.insert("line_"+l.identifier+"_end_y");
        }
    }
    return sq;
}

void CoordinateScene::mark_data_unchanged() { construction.mark_unchanged(); }
void CoordinateScene::change_data() { /*construction.update();*/ }
bool CoordinateScene::check_if_data_changed() const { return construction.has_been_updated_since_last_scene_query(); }
void CoordinateScene::on_end_transition_extra_behavior(const TransitionType tt) {
    // TODO make this micro or macroblock based
    construction.set_all_old();
}
