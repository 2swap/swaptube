#pragma once

#include "../Common/CoordinateScene.h"

class InnerBilliardsScene : public CoordinateScene {
public:
    InnerBilliardsScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    void draw_cue_stick();
    void draw_ball();
    void draw_trail();
};
