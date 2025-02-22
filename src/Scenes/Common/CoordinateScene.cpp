#pragma once

#include "../Scene.cpp"
#include <vector>

string truncate_tick(double value) {
    if(abs(value) < 0.00000000001) return "0";

    // Convert double to string with a stream
    ostringstream oss;
    oss << value;
    string str = oss.str();

    // Find the position of the decimal point
    size_t decimalPos = str.find('.');

    // If there's no decimal point, just return the string
    if (decimalPos == string::npos) {
        return str;
    }

    // Remove trailing zeros
    size_t endPos = str.find_last_not_of('0');

    // If the last non-zero character is the decimal point, remove it too
    if (endPos == decimalPos) {
        endPos--;
    }

    // Create a substring up to the correct position
    return str.substr(0, endPos + 1);
}

class CoordinateScene : public Scene {
public:
    int circles_to_render = 0;
    CoordinateScene(const double width = 1, const double height = 1)
        : Scene(width, height) {
        state_manager.add_equation("left_x"   , "<center_x> .5 <zoom_x> / -");
        state_manager.add_equation("right_x"  , "<center_x> .5 <zoom_x> / +");
        state_manager.add_equation("top_y"    , "<center_y> .5 <zoom_y> / -");
        state_manager.add_equation("bottom_y" , "<center_y> .5 <zoom_y> / +");
        state_manager.add_equation("ticks_opacity", "1");
        state_manager.add_equation("circles_opacity", "0");
        state_manager.add_equation("center_x", "0");
        state_manager.add_equation("center_y", "0");
        state_manager.add_equation("zoom", "1");
        state_manager.add_equation("zoom_x", "<zoom> <w> <VIDEO_WIDTH> * / <h> <VIDEO_HEIGHT> * *");
        state_manager.add_equation("zoom_y", "<zoom>");
    }

    pair<int, int> point_to_pixel(const pair<double, double>& p) {
        const int w = get_width();
        const int h = get_height();
        const double rx = state["right_x"];
        const double ty = state["top_y"];
        const double lx = state["left_x"];
        const double by = state["bottom_y"];
        return make_pair(
            w*   (p. first-lx)/(rx-lx),
            h*(1-(p.second-ty)/(by-ty))
        );
    }

    // This is not used here, but it is used in some classes which inherit from CoordinateScene
    void draw_trail(const vector<pair<double, double>>& trail, const int trail_color, const double trail_opacity) {
        if(trail.size() == 0) return;
        if(trail_opacity < 0.01) return;
        int line_width = get_geom_mean_size()/350;
        for(int i = 0; i < trail.size()-1; i++) {
            const pair<double, double> last_point = trail[i];
            const pair<double, double> next_point = trail[i+1];
            const pair<int, int> last_pixel = point_to_pixel(last_point);
            const pair<int, int> next_pixel = point_to_pixel(next_point);
            pix.bresenham(last_pixel.first, last_pixel.second, next_pixel.first, next_pixel.second, trail_color, trail_opacity, line_width);
        }
    }

    void draw_point(const pair<double, double> point, int point_color, double point_opacity) {
        const pair<int, int> pixel = point_to_pixel(point);
        pix.fill_circle(pixel.first, pixel.second, get_geom_mean_size()/100., point_color, point_opacity);
    }

    void draw() override {
        draw_circles();
        draw_axes();
    }

    void draw_circles() {
        const double z = state["zoom_x"] + 0.0001;
        const double opa = state["circles_opacity"];
        if(opa < 0.01) return;
        const double w = get_geom_mean_size();
        for(int i = 0; i < circles_to_render; i++){
            const double x = state["circle"+to_string(i)+"_x"];
            const double y = state["circle"+to_string(i)+"_y"];
            const double r = state["circle"+to_string(i)+"_r"];
            const pair<int, int> pixel = point_to_pixel(make_pair(x,y));
            pix.fill_donut(pixel.first, pixel.second, w*r*z*0.9, w*r*z*1.1, 0xffff0000, opa);
        }
    }

    void draw_axes() {
        draw_one_axis(true);
        draw_one_axis(false);
    }

    void draw_one_axis(bool ymode) {
        const double ticks_opacity = state["ticks_opacity"];
        if(ticks_opacity < 0.01) return;
        const int w = get_width();
        const int h = get_height();
        const double gmsz = get_geom_mean_size();

        string x_y_str = ymode?"y":"x";
        const double z = state["zoom_"+x_y_str] + 0.00000001;
        const double upper_bound = ymode?state["bottom_y"]:state["right_x"];
        const double lower_bound = ymode?state[   "top_y"]:state[ "left_x"];
        const double log10z = log10(z);
        int order_mag = floor(-log10z);
        const double log_decimal = log10z-floor(log10z);
        bool not_fiveish = log_decimal < .5;
        const double interpolator = (log_decimal >= .5 ? -1 : 0) + log_decimal * 2;
        unordered_set<string> done_numbers;
        for(int d_om = 0; d_om < 2; d_om++){
            double increment = pow(10, order_mag) * (not_fiveish ? 1 : 0.5);
            for(double x_y = floor(lower_bound/increment)*increment; x_y < upper_bound; x_y += increment) {
                string truncated = truncate_tick(x_y);
                if(done_numbers.find(truncated) != done_numbers.end()) continue;
                done_numbers.insert(truncated);
                double tick_length = (d_om == 1 ? 2 * interpolator : 2) * gmsz / 128.;
                double frac = (x_y - lower_bound) / (upper_bound - lower_bound);
                if(ymode) frac = 1-frac;
                double number_opacity = d_om == 1 ? (interpolator<.5? 0 : interpolator*2-1) : 1;
                number_opacity *= 1-square(square(2.5*(.5-frac)));
                number_opacity *= ticks_opacity;
                if(number_opacity < 0) number_opacity = 0;
                int coordinate = frac * (ymode?h:w);
                if(ymode) pix.bresenham(0, coordinate, tick_length, coordinate, OPAQUE_WHITE, number_opacity, 1);
                else      pix.bresenham(coordinate, h-1, coordinate, h-1-tick_length, OPAQUE_WHITE, number_opacity, 1);
                if(number_opacity > 0){
                    ScalingParams sp(gmsz/8., gmsz/16.);
                    Pixels latex = latex_to_pix(truncated, sp);
                    if(!ymode) latex = latex.rotate_90();
                    if(ymode) pix.overlay(latex, tick_length * 1.5, coordinate - latex.h/2, number_opacity);
                    if(!ymode)pix.overlay(latex, coordinate - latex.w/2, h-1-tick_length * 1.5 - latex.h, number_opacity);
                }
            }
            if(!not_fiveish) order_mag--;
            not_fiveish = !not_fiveish;
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = {"left_x", "right_x", "top_y", "bottom_y", "zoom_x", "zoom_y", "ticks_opacity", "circles_opacity"};
        for(int i = 0; i < circles_to_render; i++) {
            sq.insert("circle"+to_string(i)+"_x");
            sq.insert("circle"+to_string(i)+"_y");
            sq.insert("circle"+to_string(i)+"_r");
        }
        return sq;
    }

    void mark_data_unchanged() override {}
    void change_data() override { }
    bool check_if_data_changed() const override { return false; }
    void on_end_transition(bool is_macroblock){}
};

