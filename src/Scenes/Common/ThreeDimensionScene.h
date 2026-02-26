#pragma once

#include <fstream>
#include "SuperScene.h"
#include <string>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <glm/gtx/string_cast.hpp>
#include "../../Host_Device_Shared/ThreeDimensionStructs.h"

struct Surface {
    glm::vec3 center;
    glm::vec3 pos_x_dir;
    glm::vec3 pos_y_dir;
    float ilr2;
    float iur2;
    string name;
    glm::vec3 normal;

    Surface(const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, const string& n);

    // constructor to make the surface fill the screen.
    Surface(const string& n);
};

struct Path {
    string name;
    vector<glm::vec3> points;
    int color;
    float opacity;
    Path(const string& n, int clr, float op = 1);
};

class ThreeDimensionScene : public SuperScene {
public:
    bool use_state_for_center;
    ThreeDimensionScene(const double width = 1, const double height = 1);

    glm::vec2 coordinate_to_pixel(glm::vec3 coordinate, bool& behind_camera);

    bool isOutsideScreen(const glm::vec2& point);

    // Utility function to find the orientation of the ordered triplet (p, q, r).
    // The function returns:
    // 1 --> Clockwise
    // 2 --> Counterclockwise
    int orientation(const glm::vec2& p, const glm::vec2& q, const glm::vec2& r);

    // Function to check if a point is on the left side of a directed line segment.
    bool isLeft(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c);

    bool isInsideConvexPolygon(const glm::vec2& point, const vector<glm::vec2>& polygon);

    bool lineSegmentsIntersect(const glm::vec2& p1, const glm::vec2& q1, const glm::vec2& p2, const glm::vec2& q2);

    // Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
    bool onSegment(const glm::vec2& p, const glm::vec2& q, const glm::vec2& r);

    bool should_render_surface(vector<glm::vec2> corners);

    virtual void render_surface(const Surface& surface);

    void set_camera_direction();

    float squaredDistance(const glm::vec3& a, const glm::vec3& b);

    void draw() override;

    const StateQuery populate_state_query() const override;

    void add_point(const Point& p);

    void add_line(const Line& l);

    void add_surface(const Surface& s, shared_ptr<Scene> sc);

    void add_surface_fade_in(const TransitionType tt, const Surface& s, shared_ptr<Scene> sc, double opa=1);

    void remove_surface(const string& name);

    void clear_lines();
    void clear_points();
    void clear_surfaces();

    glm::vec3 camera_pos;
    glm::quat camera_direction;
    glm::quat conjugate_camera_direction;
    double fov;
    double over_w_fov;
protected:
    double auto_distance;
    glm::vec3 auto_camera; // This needs to be constructed explicitly since it carries state.
    vector<Point> points;
    vector<Line> lines;
    vector<Surface> surfaces;
    map<string, Path> paths;
};
