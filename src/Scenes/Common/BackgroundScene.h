#pragma once

#include "../Scene.h"

class BackgroundScene: public Scene {
public:
    BackgroundScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
