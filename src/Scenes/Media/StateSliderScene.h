#pragma once

#include <string>
#include "../../IO/VisualMedia.h"
#include "../Scene.h"

using std::string;

string double_to_string(double value);

class StateSliderScene : public Scene {
public:
    StateSliderScene(const string& vn, const string& dn, double min_val, double max_val, const vec2& dimensions = vec2(1, 1));

    void draw() override;
    const StateQuery populate_state_query() const override;

private:
    string display_name;
    double min_value, max_value;

    void draw_slider();
};
