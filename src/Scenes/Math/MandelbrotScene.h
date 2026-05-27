#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/DevicePointer.h"

class MandelbrotScene : public CoordinateScene {
public:
    MandelbrotScene(const vec2& dimensions = vec2(1, 1));
    const StateQuery populate_state_query() const override;
    void draw() override;
private:
    DevicePointer d_pixels;
};
