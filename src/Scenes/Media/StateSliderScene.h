#pragma once

#include <string>
#include "../../IO/VisualMedia.h"
#include "../Scene.h"

using std::string;

string double_to_string(double value);

class StateSliderScene : public Scene {
public:
    StateSliderScene(const string& vn, const string& dn, double min_val, double max_val, const double width = 1, const double height = 1);

    void draw() override;
    const StateQuery populate_state_query() const override;
    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

private:
    string display_name;
    double min_value, max_value;

    void draw_slider();
};
