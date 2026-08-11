#pragma once

#include "../Scene.h"
#include "../../DataObjects/DevicePointer.h"

class FourDAlgebraScene: public Scene {
public:
    FourDAlgebraScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
};
