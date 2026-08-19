#pragma once

#include "../Common/CoordinateScene.h"
#include "../../Host_Device_Shared/OuterBilliardsShared.h"
#include <string>
#include <vector>

class OuterBilliardsScene : public CoordinateScene {
public:
    OuterBilliardsScene(const vec2& dimensions = vec2(1, 1));
    std::vector<vec2> build_orbit_path(double count);

    void draw() override;

private:
    std::vector<vec2> read_vertices();
    std::vector<vec2> orbit(const vec2& start, int steps, const std::vector<vec2>& verts, float curvature);

    float world_per_pixel();
    void draw_singularity_graph(const std::vector<vec2>& verts);
    void draw_orbit(float thickness);
};
