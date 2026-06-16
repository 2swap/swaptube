#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/DevicePointer.h"

class FourDPlaneScene: public CoordinateScene {
public:
    FourDPlaneScene(const vec2& dimensions = vec2(1, 1));
    const StateQuery populate_state_query() const override;
    void draw() override;
};
