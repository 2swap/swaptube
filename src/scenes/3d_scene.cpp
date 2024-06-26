#pragma once

#include "scene.cpp"
#include <string>
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

struct Point {
    string name;
    glm::vec3 position;
    int color; // ARGB integer representation
    double opacity = 1.0;
    Point(string n, const glm::vec3& pos, int clr) : name(n), position(pos), color(clr) {}
};

struct Line {
    string name;
    glm::vec3 position;
    glm::vec3 start;
    glm::vec3 end;
    int color; // ARGB integer representation
    double opacity = 1.0;
    Line(string n, const glm::vec3& s, const glm::vec3& e, int clr) : name(n), start(s), end(e), color(clr) {}
};

struct Surface {
    string name;
    glm::vec3 position;
    glm::vec3 center;
    glm::vec3 pos_x_dir;
    glm::vec3 pos_y_dir;
    Scene* scenePointer;
    double lr2;
    double ur2;
    double opacity = 1.0;
    Surface(string n, const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, Scene* sc) : name(n), center(c), pos_x_dir(l), pos_y_dir(u), scenePointer(sc) {
        lr2 = square(glm::length(l));
        ur2 = square(glm::length(u));
    }
};

class ThreeDimensionScene : public Scene {
public:
    ThreeDimensionScene(const int width, const int height) : Scene(width, height), sketchpad(width, height) {}
    ThreeDimensionScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), sketchpad(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    std::pair<double, double> coordinate_to_pixel(glm::vec3 coordinate, bool& behind_camera) {
        // Rotate the coordinate based on the camera's orientation
        coordinate = camera_direction * (coordinate - camera_pos) * glm::conjugate(camera_direction);
        if(coordinate.z <= 0) {behind_camera = true; return {-1000, -1000};}

        double scale = (w*fov) / coordinate.z; // perspective projection
        double x = scale * coordinate.x + w/2;
        double y = scale * coordinate.y + h/2;

        return {x, y};
    }

    glm::vec3 unproject(double px, double py) {
        // Compute the reverse of the projection
        glm::vec3 coordinate;
        coordinate.z = 10;
        double invscale = coordinate.z / (w*fov);
        coordinate.x = (px-w*.5)*invscale;
        coordinate.y = (py-h*.5)*invscale;
        coordinate = glm::conjugate(camera_direction) * coordinate * camera_direction + camera_pos;
        return coordinate;
    }

    void unit_test_unproject(){
        for(int x = 0; x < 2; x++)
            for(int y = 0; y < 2; y++){
                bool behind_camera = false;
                pair<double, double> out = coordinate_to_pixel(unproject(x, y), behind_camera);
                assert(square(out.first-x) < .01 && square(out.second-y) < .01);
            }
    }

    bool isOutsideScreen(const pair<int, int>& point, int width, int height) {
        return point.first < -width || point.first >= 2*width || point.second < -height || point.second >= 2*height;
    }

    // Utility function to find the orientation of the ordered triplet (p, q, r).
    // The function returns:
    // 0 --> p, q and r are colinear
    // 1 --> Clockwise
    // 2 --> Counterclockwise
    int orientation(const pair<int, int>& p, const pair<int, int>& q, const pair<int, int>& r) {
        int val = (q.second - p.second) * (r.first - q.first) - (q.first - p.first) * (r.second - q.second);
        if (val == 0) return 0;  // colinear
        return (val > 0) ? 1 : 2; // clock or counterclockwise
    }

    // Function to check if a point is on the left side of a directed line segment.
    bool isLeft(const pair<int, int>& a, const pair<int, int>& b, const pair<int, int>& c) {
        return ((b.first - a.first) * (c.second - a.second) - (b.second - a.second) * (c.first - a.first)) > 0;
    }

    // Function to check if a point is inside a polygon.
    bool isInsidePolygon(const pair<int, int>& point, const vector<pair<int, int>>& polygon) {
        if (polygon.size() < 3) return false; // Not a polygon

        // Extend the point to the right infinitely
        pair<int, int> extreme = {INT_MAX, point.second};

        // Count intersections of the above line with sides of polygon
        int count = 0, i = 0;
        do {
            int next = (i + 1) % polygon.size();

            // Check if the line segment from 'point' to 'extreme' intersects
            // with the line segment from 'polygon[i]' to 'polygon[next]'
            if (lineSegmentsIntersect(polygon[i], polygon[next], point, extreme)) {
                // If the point 'point' is colinear with line segment 'i-next',
                // then check if it lies on segment. If it does, return true,
                // otherwise false
                if (orientation(polygon[i], point, polygon[next]) == 0)
                   return onSegment(polygon[i], point, polygon[next]);

                count++;
            }
            i = next;
        } while (i != 0);

        // Return true if count is odd, false otherwise
        return count & 1; // Same as (count % 2 == 1)
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

        // Special Cases
        // p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;

        // p1, q1 and q2 are colinear and q2 lies on segment p1q1
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;

        // p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;

        // p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;

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
        // Check if any corner of the quadrilateral is inside the screen
        for (const auto& corner : corners) {
            if (!isOutsideScreen(corner, w, h)) {
                return true;
            }
        }

        vector<pair<int, int>> screenCorners = {{0, 0}, {w, 0}, {w, h}, {0, h}};
        for (const auto& corner : screenCorners) {
            if (isInsidePolygon(corner, corners)) {
                return true;
            }
        }

        // Check if any edge of the quadrilateral intersects with the screen edges
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            for (int j = 0; j < 4; j++) {
                int nextScreen = (j + 1) % 4;
                if (lineSegmentsIntersect(corners[i], corners[next], screenCorners[j], screenCorners[nextScreen])) {
                    return true;
                }
            }
        }

        return false;
    }

    virtual void render_surface(const Surface& surface, int padcol) {
        vector<pair<int, int>> corners(4);
        //note, ordering matters here
        bool behind_camera = false;
        corners[0] = coordinate_to_pixel(surface.center + surface.pos_x_dir + surface.pos_y_dir, behind_camera); if(behind_camera) return;
        corners[1] = coordinate_to_pixel(surface.center - surface.pos_x_dir + surface.pos_y_dir, behind_camera); if(behind_camera) return;
        corners[2] = coordinate_to_pixel(surface.center - surface.pos_x_dir - surface.pos_y_dir, behind_camera); if(behind_camera) return;
        corners[3] = coordinate_to_pixel(surface.center + surface.pos_x_dir - surface.pos_y_dir, behind_camera); if(behind_camera) return;
        //for(int i = 0; i < 4; i++) pix.fill_ellipse(corners[i].first, corners[i].second, 2, 2, WHITE);

        if(!should_render_surface(corners)) return;

        // Draw the edges of the polygon using Bresenham's function
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            sketchpad.bresenham(corners[i].first, corners[i].second, corners[next].first, corners[next].second, padcol, 1);
        }

        Pixels* p;
        surface.scenePointer->query(p);

        // Call the flood fill algorithm
        floodFillSurface(corners, surface, p, padcol); // assuming background is black and polygon is white
    }

    void floodFillSurface(vector<pair<int, int>>& corners, const Surface& surface, Pixels* p, int padcol) {
        std::queue<std::pair<int, int>> q;

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
            q.push({lerp(x1, cx, .5), lerp(y1, cy, .5)});
            //q.push({lerp(lerp(x1, x2, .5), cx, .3), lerp(lerp(y1, y2, .5), cy, .3)});
        }

        glm::vec3 normal = glm::cross(surface.pos_x_dir, surface.pos_y_dir);

        while (!q.empty()) {
            auto [cx, cy] = q.front();
            q.pop();

            // Check if the current point is within bounds and has the old color
            if (cx < 0 || cx >= w || cy < 0 || cy >= h || sketchpad.get_pixel(cx, cy) == padcol) continue;

            if(pix.get_pixel(cx, cy) == TRANSPARENT_BLACK){
                //compute position in surface's coordinate frame as a function of x and y.
                glm::vec3 particle_3d = unproject(cx, cy);
                glm::vec2 surface_coords = intersection_point(camera_pos, particle_3d-camera_pos, surface, normal);
                // Set the pixel to the new color
                double x_pix = surface_coords.x*p->w+.5;
                double y_pix = surface_coords.y*p->h+.5;
                int col = p->get_pixel(x_pix, y_pix);
                //if(p->out_of_range(x_pix, y_pix)) col = (static_cast<int>(4*x_pix/p->w) + static_cast<int>(4*y_pix/p->h)) % 2 ? WHITE : BLACK; // add tiling to void space
                col = colorlerp(TRANSPARENT_BLACK, col, surface.opacity*dag["surfaces_opacity"]);
                pix.set_pixel_with_transparency(cx, cy, col);
            }

            sketchpad.set_pixel(cx, cy, padcol);

            // Add the neighboring points to the queue
            q.push({cx + 1, cy});
            q.push({cx - 1, cy});
            q.push({cx, cy + 1});
            q.push({cx, cy - 1});
        }
    }

    void set_camera_direction() {
        camera_direction = glm::normalize(glm::quat(dag["q1"], dag["qi"], dag["qj"], dag["qk"]));
        camera_pos = glm::vec3(dag["x"], dag["y"], dag["z"]) + glm::conjugate(camera_direction) * glm::vec3(0,0,-dag["d"]) * camera_direction;
    }

    // Function to compute squared distance between two points
    float squaredDistance(const glm::vec3& a, const glm::vec3& b) {
        glm::vec3 diff = a - b;
        return glm::dot(diff, diff);
    }

    void render_3d() {
        pix.fill(TRANSPARENT_BLACK);

        if(dag["surfaces_opacity"] > 0)
            render_surfaces();
        if(dag["points_opacity"] > 0)
            render_points();
        if(dag["lines_opacity"] > 0)
            render_lines();
    }

    void render_surfaces(){
        //lots of upfront cost, so bailout if there arent any surfaces.
        if (surfaces.size() == 0) return;

        //unit_test_unproject();
        //unit_test_intersection_point();
        sketchpad.fill(BLACK);

        // Create a list of pointers to the surfaces
        std::vector<const Surface*> surfacePointers;
        for (const Surface& surface : surfaces) {
            surfacePointers.push_back(&surface);
        }

        // Sort the pointers based on distance from camera, in descending order
        std::sort(surfacePointers.begin(), surfacePointers.end(), [this](const Surface* a, const Surface* b) {
            return squaredDistance(a->center, this->camera_pos) < squaredDistance(b->center, this->camera_pos);
        });

        // Render the surfaces using the sorted pointers
        for (int i = 0; i < surfacePointers.size(); i++) {
            render_surface(*(surfacePointers[i]), i+100);
        }
        //cout << "Rendered all surfaces" << endl;
    }

    void render_points(){
        for (const Point& point : points) {
            render_point(point);
        }
    }

    void render_lines(){
        for (const Line& line : lines) {
            render_line(line);
        }
    }

    void query(bool& done_scene, Pixels*& p) override {
        set_camera_direction();
        render_3d();
        done_scene = time++ >= scene_duration_frames;
        p = &pix;
    }

    void render_point(const Point& point) {
        bool behind_camera = false;
        std::pair<int, int> pixel = coordinate_to_pixel(point.position, behind_camera);
        if(behind_camera) return;
        pix.fill_ellipse(pixel.first, pixel.second, 2, 2, colorlerp(TRANSPARENT_BLACK, point.color, dag["points_opacity"] * point.opacity));
    }

    void render_line(const Line& line) {
        bool behind_camera = false;
        std::pair<int, int> pixel1 = coordinate_to_pixel(line.start, behind_camera);
        std::pair<int, int> pixel2 = coordinate_to_pixel(line.end, behind_camera);
        if(behind_camera) return;
        pix.bresenham(pixel1.first, pixel1.second, pixel2.first, pixel2.second, colorlerp(TRANSPARENT_BLACK, line.color, dag["lines_opacity"]*line.opacity), 1);
    }

    glm::vec2 intersection_point(const glm::vec3 &particle_start, const glm::vec3 &particle_velocity, const Surface &surface, const glm::vec3 &normal) {
        // Find t
        float t = glm::dot(normal, (surface.center - particle_start)) / glm::dot(normal, particle_velocity);
        //cout << "t: " << t << endl;

        // Compute the intersection point in 3D space
        glm::vec3 intersection_3D = particle_start + t * particle_velocity;
        //cout << "intersection_3D: " << intersection_3D.x << ", " << intersection_3D.y << ", " << intersection_3D.z << endl;

        // Convert 3D intersection point to surface's local 2D coordinates
        glm::vec3 centered = intersection_3D - surface.center;
        glm::vec2 intersection_2D;
        intersection_2D.x = glm::dot(centered, surface.pos_x_dir) / surface.lr2;
        intersection_2D.y = glm::dot(centered, surface.pos_y_dir) / surface.ur2;
        //cout << "intersection_2D: " << intersection_2D.x << ", " << intersection_2D.y << endl;

        return (intersection_2D + 1.0f) * 0.5f;
    }

    void unit_test_intersection_point(){
        glm::vec3 particle_start(-1,-1,-1);
        glm::vec3 particle_velocity(3,2,1);
        Surface surface("test", glm::vec3(0,0,0), glm::vec3(-1,0,0), glm::vec3(0,-1,0), NULL);
        glm::vec3 normal(0,0,1);
        glm::vec2 expected_output(-.5,0);
        glm::vec2 actual_output = intersection_point(particle_start, particle_velocity, surface, normal);
        //cout << "actual output: " << actual_output.x << ", " << actual_output.y << endl;
        assert(glm::length(actual_output - expected_output) < 0.001);

        particle_start = glm::vec3(0,0,-1);
        particle_velocity = glm::vec3(1,10,1);
        surface = Surface("test2", glm::vec3(0,0,0), glm::vec3(-1,0,1), glm::vec3(0,-1,0), NULL);
        normal = glm::vec3(-1,0,-1);
        expected_output = glm::vec2(.25,-2);
        actual_output = intersection_point(particle_start, particle_velocity, surface, normal);
        //cout << "actual output: " << actual_output.x << ", " << actual_output.y << endl;
        assert(glm::length(actual_output - expected_output) < 0.001);
    }

    void add_point(Point point) {
        points.push_back(point);
    }

    void add_surface(Surface s) {
        surfaces.push_back(s);
    }

public:
    std::vector<Point> points;
    std::vector<Line> lines;
    std::vector<Surface> surfaces;
    glm::vec3 camera_pos;
    glm::quat camera_direction;  // Quaternion representing the camera's orientation
    double fov = .282*3/2;
    Pixels sketchpad;
};
