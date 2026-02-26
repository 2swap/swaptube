#pragma once

#include "../Common/CoordinateScene.h"
#include <vector>
#include <string>
#include <algorithm>

class LineChartScene : public CoordinateScene {
public:
    LineChartScene(const double width = 1, const double height = 1);

    float number_to_add = 0;
    bool ready_to_add = false;

    std::vector<float> data_points;

    float get_max_data_point() const;
    float get_min_data_point() const;

    void on_end_transition_extra_behavior(const TransitionType tt) override;
    void add_data_point(TransitionType tt, float y);

    void draw() override;

    void render_chart();
};
