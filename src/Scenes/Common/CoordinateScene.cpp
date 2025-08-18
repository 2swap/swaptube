#pragma once

#include "../Scene.cpp"
#include "../../DataObjects/GeometricConstruction.cpp"
#include <vector>

string truncate_tick(const float value, const bool append_i) {
    if(abs(value) < 0.00000000001) return "0";

    // Convert float to string with a stream
    ostringstream oss;
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

    if(append_i) {
        if(str == "0") return "0";
        if(str == "1") return "i";
        if(str == "-1") return "-i";
        return str+"i";
    }
    return str;
}

class CoordinateScene : public Scene {
public:
    bool complex_plane = false;
    GeometricConstruction construction;
    CoordinateScene(const float width = 1, const float height = 1)
        : Scene(width, height) {
        state_manager.set("left_x"   , "<center_x> .5 <zoom_x> / -");
        state_manager.set("right_x"  , "<center_x> .5 <zoom_x> / +");
        state_manager.set("top_y"    , "<center_y> .5 <zoom_y> / -");
        state_manager.set("bottom_y" , "<center_y> .5 <zoom_y> / +");
        state_manager.set("ticks_opacity", "1");
        state_manager.set("center_x", "0");
        state_manager.set("center_y", "0");
        state_manager.set("zoom", ".2");
        state_manager.set("zoom_x", "<zoom> <w> <VIDEO_WIDTH> * / <h> <VIDEO_HEIGHT> * *");
        state_manager.set("zoom_y", "<zoom>");
    }

    glm::vec2 point_to_pixel(const glm::vec2& p) {
        const glm::vec2 wh(get_width(), get_height());
        const glm::vec2 rx_by(state["right_x"], state["bottom_y"]);
        const glm::vec2 lx_ty(state["left_x"], state["top_y"]);
        const glm::vec2 flip(wh * (p-lx_ty)/(rx_by-lx_ty));
        return glm::vec2(flip.x, get_height()-1-flip.y);
    }

    // This is not used here, but it is used in some classes which inherit from CoordinateScene
    void draw_trail(const vector<glm::vec2>& trail, const int trail_color, const float trail_opacity) {
        if(trail.size() == 0) return;
        if(trail_opacity < 0.01) return;
        int line_width = get_geom_mean_size()/350;
        for(int i = 1; i < trail.size()-1; i++) {
            const glm::vec2 last_point = trail[i];
            const glm::vec2 next_point = trail[i+1];
            const glm::vec2 last_pixel = point_to_pixel(last_point);
            const glm::vec2 next_pixel = point_to_pixel(next_point);
            pix.bresenham(last_pixel.x, last_pixel.y, next_pixel.x, next_pixel.y, trail_color, trail_opacity, line_width);
        }
    }

    void draw_point(const glm::vec2 point, int point_color, float point_opacity) {
        const glm::vec2 pixel = point_to_pixel(point);
        pix.fill_circle(pixel.x, pixel.y, get_geom_mean_size()/100., point_color, point_opacity);
    }

    void draw() override {
        draw_axes();
        draw_construction();
    }

    void draw_construction() {
        double gm = get_geom_mean_size();
        double line_thickness = gm/200.;
        int construction_color = OPAQUE_WHITE;

        for(const GeometricLine& l : construction.lines) {
            const glm::vec2 start_pixel = point_to_pixel(l.start);
            const glm::vec2 end_pixel = point_to_pixel(l.end);
            pix.bresenham(start_pixel.x, start_pixel.y, end_pixel.x, end_pixel.y, construction_color, 1, line_thickness);
        }
        // TODO implement
        /*for(const GeometricArc& a : construction.arcs) {
            const glm::vec2 center_pixel = point_to_pixel(a.center);
            pix.arc(center_pixel.x, center_pixel.y, a.radius, a.start_angle, a.end_angle, construction_color, 1, line_thickness);
        }*/
        for(const GeometricPoint& p : construction.points) {
            const glm::vec2 position_pixel = point_to_pixel(p.position);
            pix.fill_circle(position_pixel.x, position_pixel.y, line_thickness * 2 * p.width_multiplier, construction_color);
            if(p.label != "" && p.width_multiplier > .4) {
                ScalingParams sp(line_thickness * 16, line_thickness * 16);
                Pixels latex = latex_to_pix(p.label, sp);
                pix.overlay(latex, position_pixel.x + line_thickness * 2 + latex.w/2, position_pixel.y + line_thickness * 2 + latex.h/2, 1);
            }
            //else throw runtime_error("GeometricPoint without label found in CoordinateScene::draw_construction()");
        }
    }

    void draw_axes() {
        draw_one_axis(true);
        draw_one_axis(false);
    }

    void draw_one_axis(bool ymode) {
        const float ticks_opacity = state["ticks_opacity"];
        if(ticks_opacity < 0.01) return;
        const int w = get_width();
        const int h = get_height();
        const float gmsz = get_geom_mean_size();

        string x_y_str = ymode?"y":"x";
        const float z = state["zoom_"+x_y_str] + 0.00000001;
        const float upper_bound = ymode?state["bottom_y"]:state["right_x"];
        const float lower_bound = ymode?state[   "top_y"]:state[ "left_x"];
        const float log10z = log10(z);
        int order_mag = floor(-log10z);
        const float log_decimal = log10z-floor(log10z);
        bool not_fiveish = log_decimal < .5;
        const float interpolator = (log_decimal >= .5 ? -1 : 0) + log_decimal * 2;
        unordered_set<string> done_numbers;
        for(int d_om = 0; d_om < 2; d_om++){
            float increment = pow(10, order_mag) * (not_fiveish ? 1 : 0.5);
            for(float x_y = floor(lower_bound/increment)*increment; x_y < upper_bound; x_y += increment) {
                string truncated = truncate_tick(x_y, ymode && complex_plane);
                if(done_numbers.find(truncated) != done_numbers.end()) continue;
                done_numbers.insert(truncated);
                float tick_length = (d_om == 1 ? 2 * interpolator : 2) * gmsz / 192.;
                glm::vec2 point = point_to_pixel(glm::vec2(x_y, x_y));
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
            if(!not_fiveish) order_mag--;
            not_fiveish = !not_fiveish;
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = {"left_x", "right_x", "top_y", "bottom_y", "zoom_x", "zoom_y", "ticks_opacity"};
        return sq;
    }

    void mark_data_unchanged() override { construction.mark_unchanged(); }
    void change_data() override { /*construction.update();*/ }
    bool check_if_data_changed() const override { return construction.has_been_updated_since_last_scene_query(); }
};
