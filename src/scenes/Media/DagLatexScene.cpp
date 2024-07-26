#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "../../io/visual_media.cpp"
#include "../Scene.cpp"

string double_to_string(double value, int precision) {
    ostringstream out;
    out << fixed << setprecision(precision) << value;
    string str = out.str();
    str.erase(str.find_last_not_of('0') + 1, string::npos);
    if (str.back() == '.') str.pop_back();
    return str;
}

class DagLatexScene : public Scene {
public:
    DagLatexScene(const string& vn, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : Scene(width, height), variable_name(vn) { }

    void query(Pixels*& p) override {
        ScalingParams sp(pix.w, pix.h);
        string eqn_str = latex_text(variable_name) + " = " + double_to_string((*dag)[variable_name], 4);
        Pixels equation_pixels = eqn_to_pix(eqn_str, sp);
        pix.fill(TRANSPARENT_BLACK);
        pix.overwrite(equation_pixels, 0, 0);
        p = &pix;
    }

private:
    double scale_factor;

    // Things used for non-transition states
    string variable_name;
    pair<int, int> coords;
};
