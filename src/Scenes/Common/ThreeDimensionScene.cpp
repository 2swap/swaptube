#pragma once

#include "../Scene.cpp"
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

extern "C" void cuda_render_surface(
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

enum NodeHighlightType {
    NORMAL,
    RING,
    BULLSEYE,
};

class ThreeDimensionScene;

class ThreeDimensionalObject {
public:
    glm::vec3 center;
    int color;
    float opacity;
    ThreeDimensionalObject(const glm::vec3& pos, int col, float op) : center(pos), color(col), opacity(op) {}
    virtual ~ThreeDimensionalObject() = default;
    virtual void render(ThreeDimensionScene& scene) const = 0;
};

class Point : public ThreeDimensionalObject {
public:
    NodeHighlightType highlight;
    Point(const glm::vec3& pos, int clr, NodeHighlightType hlt = NORMAL, float op = 1)
        : ThreeDimensionalObject(pos, clr, op), highlight(hlt) {}

    void render(ThreeDimensionScene& scene) const override;
};

class Line : public ThreeDimensionalObject {
public:
    glm::vec3 start;
    glm::vec3 end;
    Line(const glm::vec3& s, const glm::vec3& e, int clr, float op = 1)
        : ThreeDimensionalObject((s + e) * glm::vec3(0.5f), clr, op), start(s), end(e) {}

    void render(ThreeDimensionScene& scene) const override;
};

class Surface : public ThreeDimensionalObject {
public:
    glm::vec3 pos_x_dir;
    glm::vec3 pos_y_dir;
    shared_ptr<Scene> scenePointer;
    float ilr2;
    float iur2;
    glm::vec3 normal;

    Surface(const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, shared_ptr<Scene> sc, float op = 1)
        : ThreeDimensionalObject(c, 0, op), pos_x_dir(l), pos_y_dir(u), scenePointer(sc),
          ilr2(0.5 / square(glm::length(l))), iur2(0.5 / square(glm::length(u))), normal(glm::cross(pos_x_dir, pos_y_dir)) {}
    void render(ThreeDimensionScene& scene) const override;
};

glm::vec3 midpoint(const vector<glm::vec3>& vecs){
    glm::vec3 ret(0.0f, 0.0f, 0.0f);
    for(const glm::vec3& vec : vecs) ret += vec;
    return ret * (1.0f/vecs.size());
}

class Polygon : public ThreeDimensionalObject {
public:
    vector<glm::vec3> vertices;

    Polygon(const vector<glm::vec3>& verts, int _color)
        : ThreeDimensionalObject(midpoint(verts), _color, 1), vertices(verts) {}
    void render(ThreeDimensionScene& scene) const override;
};

class ThreeDimensionScene : public Scene {
public:
    ThreeDimensionScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), sketchpad(width, height) {
        state_manager.set(unordered_map<string, string>{
            {"fov", ".5"},
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"d", "2"},
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
        // Rotate the coordinate based on the camera's orientation
        coordinate = camera_direction * (coordinate - camera_pos) * conjugate_camera_direction;
        if(coordinate.z <= 0) {behind_camera = true; return {-1000, -1000};}

        double scale = (get_width()*fov) / coordinate.z; // perspective projection
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
        return (val > 0) ? 1 : 2; // clock or counterclockwise
    }

    // Function to check if a point is on the left side of a directed line segment.
    bool isLeft(const pair<int, int>& a, const pair<int, int>& b, const pair<int, int>& c) {
        return ((b.first - a.first) * (c.second - a.second) - (b.second - a.second) * (c.first - a.first)) > 0;
    }

    // Function to check if a point is inside a polygon.
    bool isInsideConvexPolygon(const pair<int, int>& point, const vector<pair<int, int>>& polygon) {
        if (polygon.size() < 3) return false; // Not a polygon

        // Extend the point to the right infinitely
        pair<int, int> extreme = {100000, point.second};

        for(int i = 0; i < polygon.size(); i++) {
            int next = (i + 1) % polygon.size();

            if (lineSegmentsIntersect(polygon[i], polygon[next], point, extreme)) {
                return true;
            }
        }

        return false;
    }

    // Function to check if two line segments (p1,q1) and (p2,q2) intersect.
    bool lineSegmentsIntersect(const pair<int, int>& p1, const pair<int, int>& q1, const pair<int, int>& p2, const pair<int, int>& q2) {
        // Find the four orientations needed for the general and special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);

        // General case
        if (o1 != o2 && o3 != o4)
            return true;

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

        // Attempt to skip this render if possible
        if(this_surface_opacity < .001) return;

        vector<pair<int, int>> corners(4);
        // note, ordering matters here
        bool behind_camera_1 = false, behind_camera_2 = false, behind_camera_3 = false, behind_camera_4 = false;
        corners[0] = coordinate_to_pixel(surface.center + surface.pos_x_dir + surface.pos_y_dir, behind_camera_1);
        corners[1] = coordinate_to_pixel(surface.center - surface.pos_x_dir + surface.pos_y_dir, behind_camera_2);
        corners[2] = coordinate_to_pixel(surface.center - surface.pos_x_dir - surface.pos_y_dir, behind_camera_3);
        corners[3] = coordinate_to_pixel(surface.center + surface.pos_x_dir - surface.pos_y_dir, behind_camera_4);
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
        surface.scenePointer->query(queried);

        cuda_render_surface(
            pix.pixels,
            x1, y1, plot_w, plot_h, pix.w,
            queried->pixels.data(),
            surface.scenePointer->get_width(),
            surface.scenePointer->get_height(),
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

    void mark_data_unchanged() override { }
    void change_data() override {
        for(const auto& surface : surfaces){
            surface.scenePointer->change_data();
        }
    }

    bool check_if_data_changed() const override {
        for(const auto& surface : surfaces){
            if(surface.scenePointer->check_if_data_changed()) return true;
        }
        return false;
    }

    void on_end_transition() override {
        for(const auto& surface : surfaces){
            surface.scenePointer->on_end_transition();
        }
    }

    bool has_subscene_state_changed() const override {
        for(const auto& surface : surfaces){
            if(surface.scenePointer->check_if_state_changed()) return true;
        }
        return false;
    }

    void set_camera_direction() {
        camera_direction = glm::normalize(glm::quat(state["q1"], state["qi"], state["qj"], state["qk"]));
        conjugate_camera_direction = glm::conjugate(camera_direction);
        camera_pos = glm::vec3(state["x"], state["y"], state["z"]) + conjugate_camera_direction * glm::vec3(0,0,-state["d"]) * camera_direction;
    }

    // Function to compute squared distance between two points
    float squaredDistance(const glm::vec3& a, const glm::vec3& b) {
        glm::vec3 diff = a - b;
        return glm::dot(diff, diff);
    }

    void draw() override {
        fov = state["fov"];
        over_w_fov = 1/(get_width()*fov);
        halfwidth = get_width()*.5;
        halfheight = get_height()*.5;

        set_camera_direction();
        sketchpad.fill(OPAQUE_BLACK);

        vector<ThreeDimensionalObject> objects;

        // Create a list of pointers to the things
        vector<const ThreeDimensionalObject*> obj_ptrs;
        obj_ptrs.clear();
        if (obj_ptrs.empty()) {
            for (const Surface& surface : surfaces)
                obj_ptrs.push_back(&surface);
            for (const Line& line : lines)
                obj_ptrs.push_back(&line);
            for (const Point& point : points)
                obj_ptrs.push_back(&point);
        }

        // Sort the pointers based on distance from camera
        sort(obj_ptrs.begin(), obj_ptrs.end(), [this](const ThreeDimensionalObject* a, const ThreeDimensionalObject* b) {
            return squaredDistance(a->center, this->camera_pos) > squaredDistance(b->center, this->camera_pos);
        });

        for (int i = 0; i < obj_ptrs.size(); i++) {
            obj_ptrs[i]->render(*this);
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{
            "fov", "x", "y", "z", "d", "q1", "qi", "qj",
            "qk", "surfaces_opacity", "lines_opacity", "points_opacity"
        };
    }

    void render_point(const Point& point) {
        if(point.opacity < .001 || state["points_opacity"] < .001) return;

        bool behind_camera = false;
        pair<int, int> pixel = coordinate_to_pixel(point.center, behind_camera);
        if(behind_camera) return;
        double dot_size = pix.w/400.; if(point.highlight == RING){
            pix.fill_ellipse(pixel.first, pixel.second, dot_size*2  , dot_size*2  , point.color);
            pix.fill_ellipse(pixel.first, pixel.second, dot_size*1.5, dot_size*1.5, OPAQUE_BLACK);
        } else if(point.highlight == BULLSEYE){
            pix.fill_ellipse(pixel.first, pixel.second, dot_size*3  , dot_size*3  , point.color);
            pix.fill_ellipse(pixel.first, pixel.second, dot_size*2.5, dot_size*2.5, OPAQUE_BLACK);
            pix.fill_ellipse(pixel.first, pixel.second, dot_size*2  , dot_size*2  , point.color);
            pix.fill_ellipse(pixel.first, pixel.second, dot_size*1.5, dot_size*1.5, OPAQUE_BLACK);
        }
        if(point.opacity == 0) return;
        pix.fill_ellipse(pixel.first, pixel.second, dot_size, dot_size, colorlerp(TRANSPARENT_BLACK, point.color, state["points_opacity"] * point.opacity));
    }

    void render_line(const Line& line) {
        if(line.opacity < .001 || state["lines_opacity"] < .001) return;
        bool behind_camera = false;
        pair<int, int> pixel1 = coordinate_to_pixel(line.start, behind_camera);
        pair<int, int> pixel2 = coordinate_to_pixel(line.end, behind_camera);
        if(behind_camera) return;
        pix.bresenham(pixel1.first, pixel1.second, pixel2.first, pixel2.second, line.color, state["lines_opacity"] * line.opacity, 1);
    }

    void add_point(const Point& p) {
        points.push_back(p);
        obj_ptrs.clear();
    }

    void add_line(const Line& l) {
        lines.push_back(l);
        obj_ptrs.clear();
    }

    void add_surface(const Surface& s) {
        surfaces.push_back(s);
        s.scenePointer->state_manager.set_parent(&state_manager);
        obj_ptrs.clear();
    }

    void remove_surface(shared_ptr<Scene> s) {
        for (auto it = surfaces.begin(); it != surfaces.end(); ){
            if (it->scenePointer == s){
                it->scenePointer->state_manager.set_parent(nullptr);
                it = surfaces.erase(it);
            }
            else ++it;
        }
        obj_ptrs.clear();
    }

    void clear_lines(){ lines.clear(); obj_ptrs.clear(); }
    void clear_points(){ points.clear(); obj_ptrs.clear(); }
    void clear_surfaces(){
        for (auto it = surfaces.begin(); it != surfaces.end(); ++it){
            it->scenePointer->state_manager.set_parent(nullptr);
        }
        surfaces.clear();
        obj_ptrs.clear();
    }

    glm::vec3 camera_pos;
    glm::quat camera_direction;  // Quaternion representing the camera's orientation
    glm::quat conjugate_camera_direction;
    float fov;
    float over_w_fov;
    float halfwidth;
    float halfheight;
    Pixels sketchpad;
protected:
    vector<const ThreeDimensionalObject*> obj_ptrs;
    vector<Point> points;
    vector<Line> lines;
    vector<Surface> surfaces;
};

void   Point::render(ThreeDimensionScene& scene) const { scene.render_point  (*this); }
void    Line::render(ThreeDimensionScene& scene) const { scene.render_line   (*this); }
void Surface::render(ThreeDimensionScene& scene) const { scene.render_surface(*this); }
// To be implemented. void Polygon::render(ThreeDimensionScene& scene) const { scene.render_polygon(*this); }
