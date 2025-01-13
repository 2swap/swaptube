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
    CoordinateScene(const double width = 1, const double height = 1)
        : Scene(width, height) {
        state_manager.add_equation("left_x"  , "<center_x> .5 <zoom> / -");
        state_manager.add_equation("right_x" , "<center_x> .5 <zoom> / +");
        state_manager.add_equation("top_y"   , "<center_y> .5 <zoom> / -");
        state_manager.add_equation("bottom_y", "<center_y> .5 <zoom> / +");
    }

    void draw() override {
        render_axes();
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
        for(int d_om = 0; d_om < 2; d_om++){
            double increment = pow(10, order_mag) * (not_fiveish ? 1 : 0.5);
            for(double x = floor(lx/increment)*increment; x < rx; x += increment) {
                double modified_interpolator = interpolator;
                if(static_cast<int>(round(x / increment)) % 10 == 0) {
                    if(d_om == 1) continue;
                    modified_interpolator = 1;
                }
                double tick_length = (1+modified_interpolator-d_om) *5; 
                double number_opacity = d_om == 1 ? (modified_interpolator<.5? 0 : modified_interpolator*2-1) : 1;
                double frac = (x - lx) / (rx - lx);
                int x_pix = frac * w;
                pix.bresenham(x_pix, 0, x_pix, tick_length, OPAQUE_WHITE, 1, 1);
                ScalingParams sp(40, 20);
                if(number_opacity > 0){
                    Pixels latex = latex_to_pix(truncate_tick(x), sp);
                    pix.overlay(latex, x_pix - latex.w/2, tick_length * 1.5, number_opacity);
                }
            }
            for(double y = floor(ty/increment)*increment; y < by; y += increment) {
                double modified_interpolator = interpolator;
                if(static_cast<int>(round(y / increment)) % 10 == 0) {
                    if(d_om == 1) continue;
                    modified_interpolator = 1;
                }
                double tick_length = (1+modified_interpolator-d_om) *5; 
                double number_opacity = d_om == 1 ? (modified_interpolator<.5? 0 : modified_interpolator*2-1) : 1;
                double frac = 1 - (y - ty) / (by - ty);
                int y_pix = frac * h;
                pix.bresenham(0, y_pix, tick_length, y_pix, OPAQUE_WHITE, 1, 1);
                ScalingParams sp(40, 20);
                if(number_opacity > 0){
                    Pixels latex = latex_to_pix(truncate_tick(y), sp);
                    pix.overlay(latex, tick_length * 1.5, y_pix - latex.h/2, number_opacity);
                }
            }
            if(!not_fiveish) order_mag--;
            not_fiveish = !not_fiveish;
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"left_x", "right_x", "top_y", "bottom_y", "zoom"};
    }

    void mark_data_unchanged() override {}
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }
    void on_end_transition(){}
};

