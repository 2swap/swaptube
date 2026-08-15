#pragma once
#include "../../DataObjects/Pendulum.h"
#include "../Common/CoordinateScene.h"
#include <math.h>

class MovingPendulumGridScene : public CoordinateScene {
public:
    MovingPendulumGridScene(const vec2& dimensions = vec2(1, 1));

    void draw_grid();
    void draw() override;
};
