#pragma once
#include "../../DataObjects/Pendulum.h"
#include "../Common/CoordinateScene.h"
#include <math.h>

class MovingPendulumGridScene : public CoordinateScene {
public:
    MovingPendulumGridScene(const vec2& dimensions = vec2(1, 1));

    const StateQuery populate_state_query() const override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    void draw_grid();
    void draw() override;
};
