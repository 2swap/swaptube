#pragma once

#include "../Scene.h"

class ThreeDAlgebraScene: public Scene {
public:
    ThreeDAlgebraScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
