#pragma once

#include "SuperScene.cpp"
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
#include "ThreeDimensionStructs.cpp"

extern "C" {
    void render_points_on_gpu(
        unsigned int* h_pixels, int width, int height,
        float geom_mean_size, float points_opacity,
        Point* h_points, int num_points,
        glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov);

    void render_lines_on_gpu(
        unsigned int* h_pixels, int width, int height,
        float geom_mean_size, int thickness, float lines_opacity,
        Line* h_lines, int num_lines,
        glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov);

    void cuda_render_surface(
        vector<unsigned int>& pix,
        int x1,
        int y1,
        int plot_w,
        int plot_h,
        int pixels_w,
        unsigned int* d_surface,
        int surface_w,
        int surface_h,
        float opacity,
        glm::vec3 camera_pos,
        glm::quat camera_direction,
        glm::quat conjugate_camera_direction,
        const glm::vec3& surface_normal,
        const glm::vec3& surface_center,
        const glm::vec3& surface_pos_x_dir,
        const glm::vec3& surface_pos_y_dir,
        const float surface_ilr2,
        const float surface_iur2,
        float halfwidth,
        float halfheight,
        float over_w_fov);
}

struct Surface {
    glm::vec3 center;
    float opacity;
    glm::vec3 pos_x_dir;
    glm::vec3 pos_y_dir;
    float ilr2;
    float iur2;
    string name;
    glm::vec3 normal;

    Surface(const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, const string& n, float op = 1)
        : center(c), opacity(op), pos_x_dir(l),
          pos_y_dir(u*(VIDEO_WIDTH/static_cast<float>(VIDEO_HEIGHT))),
          name(n) {
        ilr2 = 0.5f / (l.x*l.x + l.y*l.y + l.z*l.z);
        iur2 = 0.5f / (u.x*u.x + u.y*u.y + u.z*u.z);
        normal = glm::cross(pos_x_dir, pos_y_dir);
    }
};

class ThreeDimensionScene : public SuperScene {
public:
    bool use_state_for_center;
    ThreeDimensionScene(const double width = 1, const double height = 1)
        : SuperScene(width, height), use_state_for_center(false), sketchpad(width, height) {
        state_manager.set(unordered_map<string, string>{
            {"fov", ".5"},
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"d", "1"},
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "0"},
            {"surfaces_opacity", "1"},
            {"lines_opacity", "1"},
            {"points_opacity", "1"}
        });
    }

    pair<double, double> coordinate_to_pixel(glm::vec3 coordinate, bool& behind_camera) {
        coordinate = camera_direction * (coordinate - camera_pos) * conjugate_camera_direction;
        if(coordinate.z <= 0) {behind_camera = true; return {-1000, -1000};}

        double scale = (get_geom_mean_size()*fov) / coordinate.z;
        double x = scale * coordinate.x + get_width()/2;
        double y = scale * coordinate.y + get_height()/2;
        return {x, y};
    }

    bool isOutsideScreen(const pair<int, int>& point) {
        return point.first < 0 || point.first >= get_width() || point.second < 0 || point.second >= get_height();
    }

    // Utility function to find the orientation of the ordered triplet (p, q, r).
    // The function returns:
    // 1 --> Clockwise
    // 2 --> Counterclockwise
    int orientation(const pair<int, int>& p, const pair<int, int>& q, const pair<int, int>& r) {
        int val = (q.second - p.second) * (r.first - q.first) - (q.first - p.first) * (r.second - q.second);
        return (val > 0) ? 1 : 2; // clockwise or counterclockwise
    }

    // Function to check if a point is on the left side of a directed line segment.
    bool isLeft(const pair<int, int>& a, const pair<int, int>& b, const pair<int, int>& c) {
        return ((b.first - a.first) * (c.second - a.second) - (b.second - a.second) * (c.first - a.first)) > 0;
    }

    bool isInsideConvexPolygon(const pair<int, int>& point, const vector<pair<int, int>>& polygon) {
        if (polygon.size() < 3) return false; // Not a polygon

        // Extend the point to the right infinitely
        pair<int, int> extreme = {100000, point.second};
        for (int i = 0; i < polygon.size(); i++) {
            int next = (i + 1) % polygon.size();
            if (lineSegmentsIntersect(polygon[i], polygon[next], point, extreme)) {
                return true;
            }
        }
        return false;
    }

    bool lineSegmentsIntersect(const pair<int, int>& p1, const pair<int, int>& q1, const pair<int, int>& p2, const pair<int, int>& q2) {
        // Find the four orientations needed for the general and special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);
        if (o1 != o2 && o3 != o4)
            return true; // General case
        return false; // Doesn't fall in any of the above cases
    }

    // Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
    bool onSegment(const pair<int, int>& p, const pair<int, int>& q, const pair<int, int>& r) {
        if (q.first <= max(p.first, r.first) && q.first >= min(p.first, r.first) &&
            q.second <= max(p.second, r.second) && q.second >= min(p.second, r.second))
            return true;
        return false;
    }

    bool should_render_surface(vector<pair<int, int>> corners){
        // We identify that polygons A and B share some amount of area, if and only if any one of these 3 conditions are met:
        // 1. A corner of A is inside B
        // 2. A corner of B is inside A
        // 3. There is an edge a in A and an edge b in B such that a and b intersect.

        // Check if any corner of the quadrilateral is inside the screen
        for (const auto& corner : corners)
            if (!isOutsideScreen(corner))
                return true;

        int w = get_width();
        int h = get_height();
        vector<pair<int, int>> screenCorners = {{0, 0}, {w, 0}, {w, h}, {0, h}};
        for (const auto& corner : screenCorners)
            if (isInsideConvexPolygon(corner, corners))
                return true;

        // Check if any edge of the quadrilateral intersects with the screen edges
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (lineSegmentsIntersect(corners[i], corners[(i + 1) % 4], screenCorners[j], screenCorners[(j + 1) % 4]))
                    return true;
        return false;
    }

    virtual void render_surface(const Surface& surface) {
        float this_surface_opacity = surface.opacity * state["surfaces_opacity"];

        glm::vec3 surface_center = surface.center;
        if(use_state_for_center) surface_center = glm::vec3(state[surface.name + ".x"], state[surface.name + ".y"], state[surface.name + ".z"]);
        if(this_surface_opacity < .001) return;

        vector<pair<int, int>> corners(4);
        bool behind_camera_1 = false, behind_camera_2 = false, behind_camera_3 = false, behind_camera_4 = false;
        corners[0] = coordinate_to_pixel(surface_center + surface.pos_x_dir + surface.pos_y_dir, behind_camera_1);
        corners[1] = coordinate_to_pixel(surface_center - surface.pos_x_dir + surface.pos_y_dir, behind_camera_2);
        corners[2] = coordinate_to_pixel(surface_center - surface.pos_x_dir - surface.pos_y_dir, behind_camera_3);
        corners[3] = coordinate_to_pixel(surface_center + surface.pos_x_dir - surface.pos_y_dir, behind_camera_4);
        if(behind_camera_1 && behind_camera_2 && behind_camera_3 && behind_camera_4) return;
        if(!should_render_surface(corners)) return;

        int x1 = numeric_limits<int>::max();
        int y1 = numeric_limits<int>::max();
        int x2 = numeric_limits<int>::min();
        int y2 = numeric_limits<int>::min();

        for(const auto& corner : corners){
            x1 = min(x1, corner.first );
            y1 = min(y1, corner.second);
            x2 = max(x2, corner.first );
            y2 = max(y2, corner.second);
        }
        x1 = max(x1, 0);
        y1 = max(y1, 0);
        x2 = min(x2, get_width()-1);
        y2 = min(y2, get_height()-1);
        int plot_w = x2 - x1 + 1;
        int plot_h = y2 - y1 + 1;

        Pixels* queried = NULL;
        subscenes[surface.name]->query(queried);

        cuda_render_surface(
            pix.pixels,
            x1, y1, plot_w, plot_h, pix.w,
            queried->pixels.data(),
            subscenes[surface.name]->get_width(),
            subscenes[surface.name]->get_height(),
            this_surface_opacity,
            camera_pos,
            camera_direction,
            conjugate_camera_direction,
            surface.normal,
            surface.center,
            surface.pos_x_dir,
            surface.pos_y_dir,
            surface.ilr2,
            surface.iur2,
            halfwidth,
            halfheight,
            over_w_fov
        );
    }

    void set_camera_direction() {
        camera_direction = glm::normalize(glm::quat(state["q1"], state["qi"], state["qj"], state["qk"]));
        conjugate_camera_direction = glm::conjugate(camera_direction);
        float dist_to_use = (auto_distance > 0 ? max(1.0, auto_distance) : 1)*state["d"];
        glm::vec3 camera_to_use = auto_distance > 0 ? auto_camera : glm::vec3(state["x"], state["y"], state["z"]);
        camera_pos = camera_to_use + conjugate_camera_direction * glm::vec3(0,0,-dist_to_use) * camera_direction;
    }

    float squaredDistance(const glm::vec3& a, const glm::vec3& b) {
        glm::vec3 diff = a - b;
        return glm::dot(diff, diff);
    }

    void draw() override {
        fov = state["fov"];
        over_w_fov = 1/(get_geom_mean_size()*fov);
        halfwidth = get_width()*0.5f;
        halfheight = get_height()*0.5f;

        set_camera_direction();
        sketchpad.fill(OPAQUE_BLACK);

        // Render surfaces via their CUDA integration.
        for (const Surface& surface : surfaces)
            render_surface(surface);

        // TODO profile the time each of these take, see which is worse
        if (!points.empty() && state["points_opacity"] > .001) {
            render_points_on_gpu(
                pix.pixels.data(),
                get_width(),
                get_height(),
                get_geom_mean_size(),
                state["points_opacity"],
                points.data(),
                static_cast<int>(points.size()),
                camera_direction,
                camera_pos,
                conjugate_camera_direction,
                fov
            );
        }
        if (!lines.empty() && state["lines_opacity"] > .001) {
            int thickness = static_cast<int>(get_geom_mean_size() / 640.0);
            render_lines_on_gpu(
                pix.pixels.data(),
                get_width(),
                get_height(),
                get_geom_mean_size(),
                thickness,
                state["lines_opacity"],
                lines.data(),
                static_cast<int>(lines.size()),
                camera_direction,
                camera_pos,
                conjugate_camera_direction,
                fov
            );
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = SuperScene::populate_state_query();
        StateQuery extras{
            "fov","x", "y", "z", "d", "q1", "qi", "qj",
            "qk", "surfaces_opacity", "lines_opacity", "points_opacity"
        };
        for(const string& x : extras) sq.insert(x);
        if(use_state_for_center) {
            for(const Surface& surface : surfaces){
                sq.insert(surface.name + ".x");
                sq.insert(surface.name + ".y");
                sq.insert(surface.name + ".z");
            }
        }
        return sq;
    }

    void add_point(const Point& p) {
        points.push_back(p);
    }

    void add_line(const Line& l) {
        lines.push_back(l);
    }

    void add_surface(const Surface& s, shared_ptr<Scene> sc) {
        surfaces.push_back(s);
        add_subscene_check_dupe(s.name, sc);
    }

    void remove_surface(const string& name) {
        remove_subscene(name);
        for (auto it = surfaces.begin(); it != surfaces.end(); ){
            if (it->name == name){
                it = surfaces.erase(it);
            }
            else ++it;
        }
    }

    void clear_lines(){ lines.clear(); }
    void clear_points(){ points.clear(); }
    void clear_surfaces(){
        remove_all_subscenes();
        surfaces.clear();
    }

    glm::vec3 camera_pos;
    glm::quat camera_direction;
    glm::quat conjugate_camera_direction;
    double fov;
    double over_w_fov;
    double halfwidth;
    double halfheight;
    Pixels sketchpad;
protected:
    double auto_distance = -1;
    glm::vec3 auto_camera = glm::vec3(0,0,0); // This needs to be constructed explicitly since it carries state.
    vector<Point> points;
    vector<Line> lines;
    vector<Surface> surfaces;
};
