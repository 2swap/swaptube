#pragma once

#include "../Common/CoordinateScene.h"

class TwoDAlgebraScene: public CoordinateScene {
public:
    TwoDAlgebraScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
    void change_data() override;

private:
    ivec2 get_drag_pixel();
    vec2 get_diagram_point_pos(const string& name);
};
