#pragma once

#include "scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>
#include <queue>
#include <algorithm>

glm::quat PITCH_DOWN (0, 1 , 0, 0 );
glm::quat PITCH_UP   (0, -1, 0, 0 );
glm::quat   YAW_RIGHT(0, 0, -1, 0 );
glm::quat   YAW_LEFT (0, 0, 1 , 0 );
glm::quat  ROLL_CW   (0, 0 , 0, -1);
glm::quat  ROLL_CCW  (0, 0 , 0, 1 );

struct Point {
    glm::vec3 position;
    int color; // ARGB integer representation
    Point(const glm::vec3& pos, int clr) : position(pos), color(clr) {}
};

struct Line {
    glm::vec3 start;
    glm::vec3 end;
    int color; // ARGB integer representation
    Line(const glm::vec3& s, const glm::vec3& e, int clr) : start(s), end(e), color(clr) {}
};

struct Surface {
    glm::vec3 center;
    glm::vec3 left_relative;
    glm::vec3 up_relative;
    Scene* scenePointer;
    Surface(const glm::vec3& c, const glm::vec3& l, const glm::vec3& u, Scene* sc) : center(c), left_relative(l), up_relative(u), scenePointer(sc) {}
};

class ThreeDimensionScene : public Scene {
public:
    ThreeDimensionScene(const int width, const int height) : Scene(width, height), sketchpad(width, height) {init_camera();}
    ThreeDimensionScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), sketchpad(VIDEO_WIDTH, VIDEO_HEIGHT) {init_camera();}

    void init_camera(){
        camera_direction = glm::quat(1,0,0,0);
    }

    std::pair<double, double> coordinate_to_pixel(glm::vec3 coordinate) {
        // Rotate the coordinate based on the camera's orientation
        coordinate = camera_direction * (coordinate - camera_pos) * glm::conjugate(camera_direction);
        if(coordinate.z >= 0) return {-1000, -1000};

        double scale = w*.6 / coordinate.z; // perspective projection
        return {scale*coordinate.x+w/2, scale*coordinate.y+h/2};
    }

    glm::vec3 unproject(double px, double py) {
        // Compute the reverse of the projection
        glm::vec3 coordinate;
        coordinate.z = -10;
        double invscale = -16.66666/w;
        coordinate.x = (px-w*.5)*invscale;
        coordinate.y = (py-h*.5)*invscale;
        coordinate = glm::conjugate(camera_direction) * coordinate * camera_direction + camera_pos;
        return coordinate;
    }

    void unit_test_unproject(){
        for(int x = 0; x < 2; x++)
        for(int y = 0; y < 2; y++){
            pair<double, double> out = coordinate_to_pixel(unproject(x, y));
            assert(square(out.first-x) < .01 && square(out.second-y) < .01);
        }
    }

    void render_surface(const Surface& surface, int padcol) {
        std::vector<std::pair<int, int>> pointy(4);
        //note, ordering matters here
        pointy[0] = coordinate_to_pixel(surface.center + surface.left_relative + surface.up_relative);
        pointy[1] = coordinate_to_pixel(surface.center - surface.left_relative + surface.up_relative);
        pointy[2] = coordinate_to_pixel(surface.center - surface.left_relative - surface.up_relative);
        pointy[3] = coordinate_to_pixel(surface.center + surface.left_relative - surface.up_relative);
        
        // Draw the edges of the polygon using Bresenham's function
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            sketchpad.bresenham(pointy[i].first, pointy[i].second, pointy[next].first, pointy[next].second, padcol);
        }

        // Get the seed point for flood fill (average of corner points)
        int seedX = (pointy[0].first + pointy[1].first + pointy[2].first + pointy[3].first) / 4;
        int seedY = (pointy[0].second + pointy[1].second + pointy[2].second + pointy[3].second) / 4;

        Pixels* p;
        surface.scenePointer->query(p);

        // Call the flood fill algorithm
        floodFillSurface(seedX, seedY, surface, p, padcol); // assuming background is black and polygon is white
    }

    void floodFillSurface(int x, int y, const Surface& surface, Pixels* p, int padcol) {
        std::queue<std::pair<int, int>> q;
        q.push({x, y});

        glm::vec3 normal = glm::normalize(glm::cross(surface.left_relative, surface.up_relative));
        double lr2 = square(glm::length(surface.left_relative));
        double ur2 = square(glm::length(surface.up_relative));

        while (!q.empty()) {
            auto [cx, cy] = q.front();
            q.pop();

            // Check if the current point is within bounds and has the old color
            if (cx < 0 || cx >= w || cy < 0 || cy >= h || sketchpad.get_pixel(cx, cy) == padcol) continue;

            //compute position in surface's coordinate frame as a function of x and y.
            glm::vec3 particle_velocity = unproject(cx, cy);
            glm::vec2 surface_coords = intersectionPoint(particle_velocity-camera_pos, surface, normal, lr2, ur2);

            // Set the pixel to the new color
            pix.set_pixel_with_transparency(cx, cy, p->get_pixel(surface_coords.x*p->w, surface_coords.y*p->h));
            sketchpad.set_pixel(cx, cy, padcol);

            // Add the neighboring points to the queue
            q.push({cx + 1, cy});
            q.push({cx - 1, cy});
            q.push({cx, cy + 1});
            q.push({cx, cy - 1});
        }
    }

    void set_camera_pos(glm::vec3 position) {
        camera_pos = position;
        rendered = false;
    }

    void set_quat(glm::quat orientation) {
        camera_direction = glm::normalize(orientation);
        rendered = false;
    }

    void update_variables(const std::unordered_map<string, double>& variables) {
        set_camera_pos(
            glm::vec3(variables.at("x"), variables.at("y"), variables.at("z"))
        );
        set_quat(
            glm::quat(variables.at("q1"), variables.at("q2"), variables.at("q3"), variables.at("q4"))
        );
        rendered = false;
    }

    // Function to compute squared distance between two points
    float squaredDistance(const glm::vec3& a, const glm::vec3& b) {
        glm::vec3 diff = a - b;
        return glm::dot(diff, diff);
    }

    void render_3d() {
        if (random() < .01) unit_test_unproject();
        sketchpad.fill(BLACK);
        pix.fill(rand()%100==1);
        //surfaces first

        // Create a list of pointers to the surfaces
        std::vector<const Surface*> surfacePointers;
        for (const Surface& surface : surfaces) {
            surfacePointers.push_back(&surface);
        }

        // Sort the pointers based on distance from camera, in descending order
        std::sort(surfacePointers.begin(), surfacePointers.end(), [this](const Surface* a, const Surface* b) {
            return squaredDistance(a->center, this->camera_pos) > squaredDistance(b->center, this->camera_pos);
        });

        // Render the surfaces using the sorted pointers
        for (int i = 0; i < surfacePointers.size(); i++) {
            render_surface(*(surfacePointers[i]), i+100);
        }

        for (const Point& point : points) {
            render_point(point);
        }
        for (const Line& line : lines) {
            render_line(line);
        }
    }

    void query(bool& done_scene, Pixels*& p) override {
        if (!rendered) {
            render_3d();
            rendered = true;
        }
        done_scene = time++ >= scene_duration_frames;
        p = &pix;
    }

    void render_point(const Point& point) {
        std::pair<int, int> pixel = coordinate_to_pixel(point.position);
        pix.fill_ellipse(pixel.first, pixel.second, 2, 2, point.color);
    }

    void render_line(const Line& line) {
        std::pair<int, int> pixel1 = coordinate_to_pixel(line.start);
        std::pair<int, int> pixel2 = coordinate_to_pixel(line.end);
        pix.bresenham(pixel1.first, pixel1.second, pixel2.first, pixel2.second, line.color);
    }

    glm::vec2 intersectionPoint(const glm::vec3 &particle_velocity, const Surface &surface, const glm::vec3 &normal, double lr2, double ur2) {

        // Find t
        float t = glm::dot(normal, (surface.center - camera_pos)) / glm::dot(normal, particle_velocity);

        // Compute the intersection point in 3D space
        glm::vec3 intersection_3D = camera_pos + t * particle_velocity;

        // Convert 3D intersection point to surface's local 2D coordinates
        glm::vec3 gah = intersection_3D - surface.center;
        glm::vec2 intersection_2D;
        intersection_2D.x = glm::dot(gah, surface.left_relative) / lr2;
        intersection_2D.y = glm::dot(gah, surface.up_relative) / ur2;

        return (intersection_2D + 1.0f) * 0.5f;  // Mapping from [-1, 1] to [0, 1]
    }

    void add_point(Point point) {
        points.push_back(point);
    }

    void add_surface(Surface s) {
        surfaces.push_back(s);
    }

    glm::quat get_quat() {
        return camera_direction;
    }

protected:
    std::vector<Point> points;
    std::vector<Line> lines;
    std::vector<Surface> surfaces;
    glm::vec3 camera_pos;
    glm::quat camera_direction;  // Quaternion representing the camera's orientation
    Pixels sketchpad;
};
