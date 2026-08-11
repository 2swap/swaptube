#pragma once

#include "../Scene.h"

class FourDVerseScene: public Scene {
public:
    FourDVerseScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
