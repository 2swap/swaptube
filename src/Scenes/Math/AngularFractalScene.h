#pragma once

#include "../Common/CoordinateScene.h"
#include <string>

class AngularFractalScene : public CoordinateScene {
private:
    const int size;

public:
    AngularFractalScene(int sz, const vec2& dimensions = vec2(1, 1));
    void draw() override;
    void draw_angular_fractal();
    const StateQuery populate_state_query() const override;
};
