#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "../../io/visual_media.cpp"
#include "../Scene.cpp"

// doubles with 4 sig figs
string double_to_string(double value) {
    ostringstream out;
    if (value == 0) {
        return "0";
    }

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
    DagLatexScene(const string& vn, const string& dn, const int col, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : Scene(width, height), variable_name(vn), display_name(dn), color(col) {
        if(color & 0xff000000 != 0xff000000)
            failout("DagLatexScene color must be opaque");
        state_query.insert(variable_name);
    }

    bool scene_requests_rerender() const override { return false; }
    void draw() override{
        ScalingParams sp(pix.w, pix.h);
        string eqn_str = latex_text(display_name) + " = " + double_to_string(state[variable_name]);
        Pixels equation_pixels = eqn_to_pix(eqn_str, sp);
        pix.fill(TRANSPARENT_BLACK);
        pix.overwrite(equation_pixels, 0, 0);
        pix.bitwise_and(color);
    }

private:
    string variable_name;
    string display_name;
    int color;
    pair<int, int> coords;
};
