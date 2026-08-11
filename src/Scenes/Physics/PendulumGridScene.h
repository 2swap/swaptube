#pragma once

#include "../../DataObjects/Pendulum.h"
#include "../Common/CoordinateScene.h"
#include <math.h>
#include <vector>
#include <list>
#include <utility>

class PendulumGridScene : public CoordinateScene {
public:
    PendulumGridScene(const vector<PendulumGrid>& pgv, const vec2& dimensions = vec2(1, 1));

    void draw_grid();

    void draw() override;

    void draw_pendulum_trail();

    bool momentum_mode = false;
private:
    vector<PendulumGrid> grids;
};
