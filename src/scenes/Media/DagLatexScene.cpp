#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "../../io/visual_media.cpp"
#include "../Scene.cpp"

// doubles with 4 sig figs
string double_to_string(double value) {
    ostringstream out;
    if (value == 0) { return "0"; }

    int exponent = static_cast<int>(floor(log10(abs(value))));
    int significant_digits = 4 - exponent - 1;

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

class DagLatexScene : public Scene {
public:
    DagLatexScene(const string& vn, const string& dn, const int col, double min_val, double max_val, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : Scene(width, height), variable_name(vn), display_name(dn), color(col), min_value(min_val), max_value(max_val) {
        if((color & 0xff000000) != 0xff000000)
            failout("DagLatexScene color must be opaque");
        state_query.insert(variable_name);
    }

    bool scene_requests_rerender() const override { return false; }
    void draw() override {
        ScalingParams sp(pix.w, pix.h);
        string eqn_str = display_name + " = " + double_to_string(state[variable_name]);
        Pixels equation_pixels = eqn_to_pix(eqn_str, sp);
        pix.fill(TRANSPARENT_BLACK);
        draw_slider();
        pix.overlay(equation_pixels, h/2, (h-equation_pixels.h)/2.); // h/2 shifts the text over horizontally a little out of the left wall
    }

private:
    string variable_name;
    string display_name;
    int color;
    double min_value, max_value;

    void draw_slider() {
        const int outer_color = 0xff444444; // Dark gray
        const int inner_color = 0xff000000; // Black
        const int  knob_color = 0xff444444; // Dark gray
        const double outer_radius = h * .5;
        const double inner_radius = h * .4;
        const double  knob_radius = h * .3;
        const double outer_margin = h*.5 - outer_radius;
        const double inner_margin = h*.5 - inner_radius;
        pix.rounded_rect(outer_margin, outer_margin, w - outer_margin * 2, h - outer_margin * 2, outer_radius, outer_color);
        pix.rounded_rect(inner_margin, inner_margin, w - inner_margin * 2, h - inner_margin * 2, inner_radius, inner_color);

        // Calculate the position of the knob based on the variable value
        double value = state[variable_name];
        double normalized_value = (value - min_value) / (max_value - min_value);

        pix.fill_circle(outer_radius + normalized_value * (w - outer_radius*2), h/2., knob_radius, knob_color);
    }
};

