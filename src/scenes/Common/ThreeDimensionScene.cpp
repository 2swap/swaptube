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
#include <glm/gtx/string_cast.hpp>

glm::quat PITCH_DOWN (0, 1 , 0, 0 );
glm::quat PITCH_UP   (0, -1, 0, 0 );
glm::quat   YAW_RIGHT(0, 0, -1, 0 );
glm::quat   YAW_LEFT (0, 0, 1 , 0 );
glm::quat  ROLL_CW   (0, 0 , 0, -1);
glm::quat  ROLL_CCW  (0, 0 , 0, 1 );

enum NodeHighlightType {
    NORMAL,
    RING,
    BULLSEYE,
};

struct Point {
    glm::vec3 position;
    int color; // ARGB integer representation
    NodeHighlightType highlight;
    double opacity;
    Point(const glm::vec3& pos, int clr, NodeHighlightType hlt=NORMAL, double op=1) : position(pos), color(clr), highlight(hlt), opacity(op) {}
};

struct Line {
    glm::vec3 start;
    glm::vec3 end;
    glm::vec3 center;
    int color; // ARGB integer representation
    double opacity;
    Line(const glm::vec3& s, const glm::vec3& e, int clr, double op=1) : start(s), end(e), center((s+e)*glm::vec3(0.5f)), color(clr), opacity(op) {}
};

struct Surface {
    glm::vec3 center;
    glm::vec3 pos_x_dir;
    glm::vec3 pos_y_dir;
    int color;
    Scene* scenePointer;
    double opacity;
    float ilr2;
    float iur2;
    glm::vec3 normal;
    // Two types of surfaces- ones which are backed by a scene, and ones which are constant color
    Surface(const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, Scene* sc)
        : center(c), pos_x_dir(l), pos_y_dir(u), scenePointer(sc), opacity(1),
        ilr2(0.5/square(glm::length(l))), iur2(0.5/square(glm::length(u))), normal(glm::cross(pos_x_dir, pos_y_dir)) {}
    Surface(const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, int col)
        : center(c), pos_x_dir(l), pos_y_dir(u), color(col), scenePointer(NULL), opacity(1),
        ilr2(0.5/square(glm::length(l))), iur2(0.5/square(glm::length(u))), normal(glm::cross(pos_x_dir, pos_y_dir)) {}
};

class ThreeDimensionScene : public Scene {
public:
    ThreeDimensionScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), sketchpad(width, height) {
        dag->add_equations(unordered_map<string, string>{
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

        double scale = (w*fov) / coordinate.z; // perspective projection
        double x = scale * coordinate.x + w/2;
        double y = scale * coordinate.y + h/2;

        return {x, y};
    }

    glm::vec2 screen_to_surface_intersection(float dotnormcam, double px, double py, const Surface &surface) {
        // Compute the ray direction from the camera through the screen point
        glm::vec3 ray_dir((px - halfwidth) * over_w_fov, (py - halfheight) * over_w_fov, 1);
        ray_dir = conjugate_camera_direction * ray_dir * camera_direction;

        // Compute the intersection point in 3D space
        float t = dotnormcam / glm::dot(surface.normal, ray_dir);

        // Convert 3D intersection point to surface's local 2D coordinates
        glm::vec3 centered = camera_pos + t * ray_dir - surface.center;
        glm::vec2 intersection_2D(
            glm::dot(centered, surface.pos_x_dir) * surface.ilr2 + 0.5f,
            glm::dot(centered, surface.pos_y_dir) * surface.iur2 + 0.5f
        );

        return intersection_2D;
    }

    bool isOutsideScreen(const pair<int, int>& point) {
        return point.first < 0 || point.first >= w || point.second < 0 || point.second >= h;
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

    virtual void render_surface(const Surface& surface, int padcol) {
        vector<pair<int, int>> corners(4);
        //note, ordering matters here
        bool behind_camera_1 = false, behind_camera_2 = false, behind_camera_3 = false, behind_camera_4 = false;
        corners[0] = coordinate_to_pixel(surface.center + surface.pos_x_dir + surface.pos_y_dir, behind_camera_1);
        corners[1] = coordinate_to_pixel(surface.center - surface.pos_x_dir + surface.pos_y_dir, behind_camera_2);
        corners[2] = coordinate_to_pixel(surface.center - surface.pos_x_dir - surface.pos_y_dir, behind_camera_3);
        corners[3] = coordinate_to_pixel(surface.center + surface.pos_x_dir - surface.pos_y_dir, behind_camera_4);
        if(behind_camera_1 && behind_camera_2 && behind_camera_3 && behind_camera_4) return;

        if(!should_render_surface(corners)) return;

        // Draw the edges of the polygon using Bresenham's function
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            sketchpad.bresenham(corners[i].first, corners[i].second, corners[next].first, corners[next].second, padcol, 1);
        }

        // Call the flood fill algorithm
        floodFillSurface(corners, surface, padcol); // assuming background is black and polygon is white
    }

    bool scene_requests_rerender() const override {
        for(const auto& surface : surfaces){
            if(surface.scenePointer->scene_requests_rerender())
                return false;
        }
        return false;
    }

    void floodFillSurface(vector<pair<int, int>>& corners, const Surface& surface, int padcol) {
        Pixels* p = NULL;
        if(surface.scenePointer != NULL) surface.scenePointer->query(p);

        queue<pair<int, int>> q;

        // 1. Compute the centroid of the quadrilateral
        double cx = 0, cy = 0;
        for (const auto& corner : corners) {
            cx += corner.first;
            cy += corner.second;
        }
        cx /= 4;
        cy /= 4;
        q.push({cx, cy});

        // 2. For each edge, perform linear interpolation to get intermediate points
        for (int i = 0; i < 4; i++) {
            int x1 = corners[i].first;
            int y1 = corners[i].second;
            int x2 = corners[(i+1)%4].first;
            int y2 = corners[(i+1)%4].second;
            float fineness = 50;
            for(float j = 0.5; j < fineness; j++){
                float jf = j/fineness;
                q.push({lerp(x1, x2, jf), lerp(y1, y2, jf)});
            }
        }

        double opacity = surface.opacity * state["surfaces_opacity"];
        float dotnormcam = glm::dot(surface.normal, (surface.center - camera_pos));

        if(p != NULL){ // If this is not a surface of constant color
            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();

                // Check if the current point is within bounds and has the old color
                if (cx < 0 || cx >= w || cy < 0 || cy >= h || sketchpad.get_pixel(cx, cy) == padcol) continue;

                //compute position in surface's coordinate frame as a function of x and y.
                glm::vec2 surface_coords = screen_to_surface_intersection(dotnormcam, cx, cy, surface);
                int color = p->get_pixel(surface_coords.x*p->w, surface_coords.y*p->h);

                // Set the pixel to the new color
                pix.overlay_pixel(cx, cy, color, opacity);

                sketchpad.set_pixel(cx, cy, padcol);

                // Add the neighboring points to the queue
                q.push({cx + 1, cy});
                q.push({cx - 1, cy});
                q.push({cx, cy + 1});
                q.push({cx, cy - 1});
            }
        } else {
            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();

                // Check if the current point is within bounds and has the old color
                if (cx < 0 || cx >= w || cy < 0 || cy >= h || sketchpad.get_pixel(cx, cy) == padcol) continue;

                // Set the pixel to the new color
                pix.overlay_pixel(cx, cy, surface.color, opacity);

                sketchpad.set_pixel(cx, cy, padcol);

                // Add the neighboring points to the queue
                q.push({cx + 1, cy});
                q.push({cx - 1, cy});
                q.push({cx, cy + 1});
                q.push({cx, cy - 1});
            }
        }
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

    void render_3d() {
        pix.fill(TRANSPARENT_BLACK);

        if(state["surfaces_opacity"] > 0 && !skip_surfaces)
            render_surfaces();
        if(state["points_opacity"] > 0)
            render_points();
        if(state["lines_opacity"] > 0)
            render_lines();
    }

    void render_surfaces(){
        //lots of upfront cost, so bailout if there arent any surfaces.
        if (surfaces.size() == 0 || state["surfaces_opacity"] == 0) return;

        sketchpad.fill(OPAQUE_BLACK);

        // Create a list of pointers to the surfaces
        vector<const Surface*> surfacePointers;
        for (const Surface& surface : surfaces)
            if(surface.opacity > 0)
                surfacePointers.push_back(&surface);

        // Sort the pointers based on distance from camera
        sort(surfacePointers.begin(), surfacePointers.end(), [this](const Surface* a, const Surface* b) {
            return squaredDistance(a->center, this->camera_pos) > squaredDistance(b->center, this->camera_pos);
        });

        // Render the surfaces using the sorted pointers
        for (int i = 0; i < surfacePointers.size(); i++) {
            render_surface(*(surfacePointers[i]), i+100);
        }
        //cout << "Rendered all surfaces" << endl;
    }

    void render_points(){
        for (const Point& point : points)
            render_point(point);
    }

    void render_lines(){
        vector<const Line*> line_ptrs;
        for (const Line& line : lines)
            if(line.opacity > 0)
                line_ptrs.push_back(&line);
        sort(line_ptrs.begin(), line_ptrs.end(), [this](const Line* a, const Line* b) {
            return squaredDistance(a->center, this->camera_pos) > squaredDistance(b->center, this->camera_pos);
        });
        for (const Line* line : line_ptrs)
            render_line(*line);
    }

    void pre_query() override {
        state_query = StateQuery{
            "fov",
            "x",
            "y",
            "z",
            "d",
            "q1",
            "qi",
            "qj",
            "qk",
            "surfaces_opacity",
            "lines_opacity",
            "points_opacity"
        };
        for(const Surface& s : surfaces){
            append_to_state_query(s.scenePointer->state_query);
        }
    }

    void draw() override {
        set_camera_direction();
        render_3d();
    }

    void render_point(const Point& point) {
        fov = state["fov"];
        over_w_fov = 1/(w*fov);
        halfwidth = w*.5;
        halfheight = h*.5;

        bool behind_camera = false;
        pair<int, int> pixel = coordinate_to_pixel(point.position, behind_camera);
        if(behind_camera) return;
        double dot_size = pix.w/500.; if(point.highlight == RING){
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
        if(line.opacity == 0) return;
        bool behind_camera = false;
        pair<int, int> pixel1 = coordinate_to_pixel(line.start, behind_camera);
        pair<int, int> pixel2 = coordinate_to_pixel(line.end, behind_camera);
        if(behind_camera) return;
        pix.bresenham(pixel1.first, pixel1.second, pixel2.first, pixel2.second, colorlerp(OPAQUE_BLACK, line.color, state["lines_opacity"] * line.opacity), 3);
    }

    void add_point(Point point) {
        points.push_back(point);
    }

    void add_surface(Surface s) {
        surfaces.push_back(s);
    }

    glm::vec3 camera_pos;
    glm::quat camera_direction;  // Quaternion representing the camera's orientation
    glm::quat conjugate_camera_direction;
    float fov;
    float over_w_fov;
    float halfwidth;
    float halfheight;
    Pixels sketchpad;
public:
    vector<Point> points;
    vector<Line> lines;
    vector<Surface> surfaces;
    bool skip_surfaces = false;
};
