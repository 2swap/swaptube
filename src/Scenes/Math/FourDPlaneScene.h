#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/DevicePointer.h"

class FourDPlaneScene: public CoordinateScene {
public:
    FourDPlaneScene(const vec2& dimensions = vec2(1, 1));
    const StateQuery populate_state_query() const override;
    void draw() override;
};


float **rotationMatrix(int rows, int cols, int axis1, int axis2, float angle);

float **matrixMult(float **A,float **B, int rows, int cols, int shared);
