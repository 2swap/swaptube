#pragma once

#include "../Common/CoordinateScene.h"
#include "../../Host_Device_Shared/OuterBilliardsShared.h"
#include <string>
#include <vector>

class OuterBilliardsVertexFlowScene : public CoordinateScene {
public:
    OuterBilliardsVertexFlowScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    std::vector<vec2> read_vertices();

    void draw_flow_field(const std::vector<vec2>& verts);
    void draw_ball();
};
