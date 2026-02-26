#pragma once

#include <vector>
#include "DataObject.h"

struct Ball {
    float x;
    float y;
    float vx;
    float vy;
};

class BouncingBalls : public DataObject {
public:
    std::vector<Ball> balls;
    int simulation_width;
    int simulation_height;
    float ball_radius;
    BouncingBalls(const int n, const int sw, const int sh, const float br);
    void iterate();
};
