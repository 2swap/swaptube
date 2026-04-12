#pragma once

#include <algorithm>
#include "../../DataObjects/BouncingBalls.h"
#include "../Common/CoordinateScene.h"

class BouncingBallScene : public CoordinateScene {
private:
    BouncingBalls balls;

public:
    BouncingBallScene(
        const int n,
        const double simulation_width = 10,
        const double simulation_height = 10,
        const vec2& dimensions = vec2(1, 1)
    );

    void draw() override;
};
