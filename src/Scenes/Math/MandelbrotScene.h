#pragma once

#include "../Common/CoordinateScene.h"

class MandelbrotScene : public CoordinateScene {
public:
    MandelbrotScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
