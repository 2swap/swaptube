#pragma once

#include "../Common/CoordinateScene.h"
#include <complex>
#include "../../DataObjects/DevicePointer.h"

class RootFractalScene : public CoordinateScene {
public:
    RootFractalScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
    const StateQuery populate_state_query() const override;
private:
    DevicePointer d_pixels;
};
