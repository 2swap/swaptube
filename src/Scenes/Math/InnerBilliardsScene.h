#pragma once

#include "../Common/CoordinateScene.h"

class InnerBilliardsScene : public CoordinateScene {
public:
    InnerBilliardsScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    float pocket_radius_px();
    float pocket_radius_world();
    void draw_table(const vector<vec2>& verts);
    void draw_cue_stick();
    void draw_ball(const vector<vec2>& verts);
    void draw_trail(const vector<vec2>& verts);
    void draw_trail();
};
