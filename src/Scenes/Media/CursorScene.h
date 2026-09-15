#pragma once

#include "../Scene.h"

class CursorScene: public Scene {
public:
    CursorScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
