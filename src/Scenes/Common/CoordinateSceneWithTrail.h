#pragma once

#include "CoordinateScene.h"
#include <list>
#include <utility>

class CoordinateSceneWithTrail : public CoordinateScene {
public:
    int trail_color = OPAQUE_WHITE;
    list<pair<vec2, int>> trail;
    CoordinateSceneWithTrail(const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

    void change_data() override;

    void clear_trail();

    bool check_if_data_changed() const override;
};
