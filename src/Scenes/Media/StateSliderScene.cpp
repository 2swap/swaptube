#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

// doubles with some number of  sig figs
string double_to_string(double value) {
    const int sig_figs = 2;
    ostringstream out;
    if (value == 0) { return "0"; }

    int exponent = static_cast<int>(floor(log10(abs(value))));
    int significant_digits = sig_figs - exponent - 1;

    out << fixed << setprecision(significant_digits) << value;
    string str = out.str();
    
    // Remove trailing zeros
    str.erase(str.find_last_not_of('0') + 1, string::npos);
    
    // Remove decimal point if it is the last character
    if (str.back() == '.') {
        str.pop_back();
    }
    
    return str;
}

class StateSliderScene : public Scene {
public:
    StateSliderScene(const string& vn, const string& dn, double min_val, double max_val, const double width = 1, const double height = 1)
    : Scene(width, height), display_name(dn), min_value(min_val), max_value(max_val) {
        state_manager.set(unordered_map<string, string>{
            {"value", vn},
        });
    }

    void draw() override {
        ScalingParams sp(pix.w, pix.h);
        string eqn_str = display_name + " = " + double_to_string(state["value"]);
        Pixels equation_pixels = latex_to_pix(eqn_str, sp);
        draw_slider();
        pix.overlay(equation_pixels, get_height()/2, (get_height()-equation_pixels.h)/2.); // h/2 shifts the text over horizontally a little out of the left wall
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"value"};
    }
    void mark_data_unchanged() override { }
    void on_end_transition() override{}
    void change_data() override { }
    bool check_if_data_changed() const override { return false; } // No DataObjects

private:
    string display_name;
    double min_value, max_value;

    void draw_slider() {
        const int h = get_height();
        const int w = get_width();
        const int outer_color = 0xff444444; // Dark gray
        const int inner_color = 0xff000000; // Black
        const int  knob_color = 0xff444444; // Dark gray
        const double outer_radius = h * .5;
        const double inner_radius = h * .4;
        const double  knob_radius = h * .3;
        const double outer_margin = h * .5 - outer_radius;
        const double inner_margin = h * .5 - inner_radius;
        pix.rounded_rect(outer_margin, outer_margin, w - outer_margin * 2, h - outer_margin * 2, outer_radius, outer_color);
        pix.rounded_rect(inner_margin, inner_margin, w - inner_margin * 2, h - inner_margin * 2, inner_radius, inner_color);

        // Calculate the position of the knob based on the variable value
        double value = state["value"];
        double normalized_value = (value - min_value) / (max_value - min_value);

        pix.fill_circle(outer_radius + normalized_value * (w - outer_radius*2), h/2., knob_radius, knob_color);
    }
};

