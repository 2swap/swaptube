#pragma once

#include "../Common/CoordinateScene.h"

class TwoDAlgebraScene: public CoordinateScene {
public:
    TwoDAlgebraScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
