#pragma once

#include "../Scene.cpp"
#include <vector>

string truncate_tick(double value) {
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
    vector<pair<double, double>> trail;
    CoordinateScene(const double width = 1, const double height = 1)
        : Scene(width, height) {
        state_manager.add_equation("left_x"  , "<center_x> .5 <zoom> / -");
        state_manager.add_equation("right_x" , "<center_x> .5 <zoom> / +");
        state_manager.add_equation("top_y"   , "<center_y> .5 <zoom> / -");
        state_manager.add_equation("bottom_y", "<center_y> .5 <zoom> / +");
    }

    pair<int, int> point_to_pixel(pair<double, double> p) {
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

    void draw_trail() {
        for(int i = 0; i < trail.size()-1; i++) {
            const pair<double, double> last_point = trail[i];
            const pair<double, double> next_point = trail[i+1];
            pair<int, int> last_pixel = point_to_pixel(last_point);
            pair<int, int> next_pixel = point_to_pixel(next_point);
            pix.bresenham(last_pixel.first, last_pixel.second, next_pixel.first, next_pixel.second, OPAQUE_WHITE, 1, 1);
        }
    }

    void draw() override {
        render_axes();
        draw_trail();
    }

    void render_axes() {
        const int w = get_width();
        const int h = get_height();
        const double z = state["zoom"] + 0.0001;
        const double rx = state["right_x"];
        const double ty = state["top_y"];
        const double lx = state["left_x"];
        const double by = state["bottom_y"];
        const double log10z = log10(z);
        int order_mag = floor(-log10z);
        const double log_decimal = log10z-floor(log10z);
        bool not_fiveish = log_decimal < .5;
        const double interpolator = (log_decimal >= .5 ? -1 : 0) + log_decimal * 2;
        unordered_set<string> done_numbers_x;
        unordered_set<string> done_numbers_y;
        for(int d_om = 0; d_om < 2; d_om++){
            double increment = pow(10, order_mag) * (not_fiveish ? 1 : 0.5);
            for(double x = floor(lx/increment)*increment; x < rx; x += increment) {
                string truncated = truncate_tick(x);
                if(done_numbers_x.find(truncated) != done_numbers_x.end()) continue;
                done_numbers_x.insert(truncated);
                double tick_length = (d_om == 1 ? 2 * interpolator : 2) * w / 128.; 
                double frac = (x - lx) / (rx - lx);
                double number_opacity = d_om == 1 ? (interpolator<.5? 0 : interpolator*2-1) : 1;
                number_opacity *= 1-square(square(2.5*(.5-frac)));
                if(number_opacity < 0) number_opacity = 0;
                int x_pix = frac * w;
                pix.bresenham(x_pix, h-1, x_pix, h-1-tick_length, OPAQUE_WHITE, number_opacity, 1);
                if(number_opacity > 0){
                    ScalingParams sp(w/12., w/24.);
                    Pixels latex = latex_to_pix(truncated, sp).rotate_90();
                    pix.overlay(latex, x_pix - latex.w/2, h-1-tick_length * 1.5 - latex.h, number_opacity);
                }
            }
            for(double y = floor(ty/increment)*increment; y < by; y += increment) {
                string truncated = truncate_tick(y);
                if(done_numbers_y.find(truncated) != done_numbers_y.end()) continue;
                done_numbers_y.insert(truncated);
                double tick_length = (d_om == 1 ? 2 * interpolator : 2) * w / 128.; 
                double frac = 1 - (y - ty) / (by - ty);
                double number_opacity = d_om == 1 ? (interpolator<.5? 0 : interpolator*2-1) : 1;
                number_opacity *= 1-square(square(2.5*(.5-frac)));
                if(number_opacity < 0) number_opacity = 0;
                int y_pix = frac * h;
                pix.bresenham(0, y_pix, tick_length, y_pix, OPAQUE_WHITE, number_opacity, 1);
                if(number_opacity > 0){
                    ScalingParams sp(w/12., w/24.);
                    Pixels latex = latex_to_pix(truncated, sp);
                    pix.overlay(latex, tick_length * 1.5, y_pix - latex.h/2, number_opacity);
                }
            }
            if(!not_fiveish) order_mag--;
            not_fiveish = !not_fiveish;
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"left_x", "right_x", "top_y", "bottom_y", "zoom", "trail_x", "trail_y"};
    }

    void mark_data_unchanged() override {}
    void change_data() override { trail.push_back(make_pair(state["trail_x"], state["trail_y"])); }
    bool check_if_data_changed() const override { return true; }
    void on_end_transition(){}
};

