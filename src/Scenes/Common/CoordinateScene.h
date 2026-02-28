#pragma once

#include "../Scene.h"
#include "../../DataObjects/GeometricConstruction.h"
#include <string>
#include <sstream>
#include <unordered_set>
#include <list>
#include <utility>

using std::string;

string float_to_pretty_string(const float value);
string truncate_tick(const float value, const bool append_i);

class CoordinateScene : public Scene {
public:
    bool complex_plane = false;
    GeometricConstruction construction;
    CoordinateScene(const float width = 1, const float height = 1);

    vec2 point_to_pixel(const vec2& p);
    vec2 pixel_to_point(const vec2& pix);

    // This is not used here, but it is used in some classes which inherit from CoordinateScene
    void draw_trail(const std::list<std::pair<vec2, int>>& trail, const float trail_opacity);

    void draw_point(const vec2 point, int point_color, float point_opacity);

    void draw() override;

    void draw_construction();

    void draw_zero_crosshair();

    void draw_axes();

    void draw_one_axis(bool ymode);

    const StateQuery populate_state_query() const override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;
    void on_end_transition_extra_behavior(const TransitionType tt) override;
};
