#pragma once

#include "../Scene.h"

class MultiplicationTableScene: public Scene {
public:
    MultiplicationTableScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
