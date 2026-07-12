#pragma once

#include "../Scene.h"
#include "../../DataObjects/DevicePointer.h"

class FourDVerseScene: public Scene {
public:
    FourDVerseScene(const vec2& dimensions = vec2(1, 1));
    const StateQuery populate_state_query() const override;
    void draw() override;
};
