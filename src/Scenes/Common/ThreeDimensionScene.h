#pragma once

#include "SuperScene.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "../../Host_Device_Shared/ThreeDimensionStructs.h"
#include "../../Host_Device_Shared/vec.h"

class ThreeDimensionScene : public SuperScene {
public:
    ThreeDimensionScene(const vec2& dimensions = vec2(1, 1));

    vec2 coordinate_to_pixel(vec3 coordinate, float& distance);

    void set_camera_direction();

    void draw() override;

    void add_point(const Point& p);

    void add_line(const Line& l);

    void clear_lines();
    void clear_points();

    unordered_map<int, string> point_names;

    vec3 camera_pos;
    quat camera_direction;
    double fov;
    double over_w_fov;
protected:
    vector<Point> read_state_points() const;

    vector<Point> points;
    vector<Line> lines;
};
