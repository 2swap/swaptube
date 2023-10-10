#pragma once

#include "scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>
#include <queue>

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
    ThreeDimensionScene(const int width, const int height) : Scene(width, height) {init();}
    ThreeDimensionScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {init();}

    void init(){
        camera_direction = glm::quat(1,0,0,0);
    }

    std::pair<double, double> coordinate_to_pixel(glm::vec3 coordinate) {
        // Rotate the coordinate based on the camera's orientation
        coordinate = camera_direction * (coordinate - camera_pos) * glm::conjugate(camera_direction);
        if(coordinate.z >= 0) return {-1000, -1000};

        double scale = w*.6 / coordinate.z; // perspective projection
        return {scale*coordinate.x+w/2, scale*coordinate.y+h/2};
    }

    glm::vec3 unproject(std::pair<double, double> pixel) {
        // Compute the reverse of the projection
        glm::vec3 coordinate;
        coordinate.z = -10;
        double scale = w*.6 / coordinate.z;
        coordinate.x = (pixel.first -w/2.)/scale;
        coordinate.y = (pixel.second-h/2.)/scale;
        coordinate = glm::conjugate(camera_direction) * coordinate * camera_direction + camera_pos;
        return coordinate;
    }

    void unit_test_unproject(){
        for(int x = 0; x < 2; x++)
        for(int y = 0; y < 2; y++){
            pair<double, double> in = make_pair(x, y);
            pair<double, double> out = coordinate_to_pixel(unproject(in));
            assert(square(out.first-in.first) < .01 && square(out.second-in.second) < .01);
        }
    }

    void render_surface(const Surface& surface) {
        unit_test_unproject();
        std::vector<std::pair<int, int>> pointy(4);
        //note, ordering matters here
        pointy[0] = coordinate_to_pixel(surface.center + surface.left_relative + surface.up_relative);
        pointy[1] = coordinate_to_pixel(surface.center - surface.left_relative + surface.up_relative);
        pointy[2] = coordinate_to_pixel(surface.center - surface.left_relative - surface.up_relative);
        pointy[3] = coordinate_to_pixel(surface.center + surface.left_relative - surface.up_relative);
        
        // Draw the edges of the polygon using Bresenham's function
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            pix.bresenham(pointy[i].first, pointy[i].second, pointy[next].first, pointy[next].second, WHITE);
        }

        // Get the seed point for flood fill (average of corner points)
        int seedX = (pointy[0].first + pointy[1].first + pointy[2].first + pointy[3].first) / 4;
        int seedY = (pointy[0].second + pointy[1].second + pointy[2].second + pointy[3].second) / 4;

        Pixels* p;
        surface.scenePointer->query(p);

        // Call the flood fill algorithm
        floodFillSurface(seedX, seedY, 0x42000000, surface, p); // assuming background is black and polygon is white
    }

    void floodFillSurface(int x, int y, int oldColor, const Surface& surface, Pixels* p) {
        cout << "ff" << endl;
        std::queue<std::pair<int, int>> q;
        q.push({x, y});

        while (!q.empty()) {
            auto [cx, cy] = q.front();
            q.pop();

            // Check if the current point is within bounds and has the old color
            if (cx < 0 || cx >= w || cy < 0 || cy >= h || pix.get_pixel(cx, cy) != 0x42000000) continue;

            // Set the pixel to the new color
            //compute position in surface's coordinate frame as a function of x and y.
            glm::vec3 particle_velocity = unproject(make_pair(cx, cy));
            glm::vec2 surface_coords = intersectionPoint(particle_velocity-camera_pos, surface);

            pix.set_pixel(cx, cy, p->get_pixel(surface_coords.x*p->w, surface_coords.y*p->h));

            // Add the neighboring points to the queue
            q.push({cx + 1, cy});
            q.push({cx - 1, cy});
            q.push({cx, cy + 1});
            q.push({cx, cy - 1});
        }
        cout << "xxs" << endl;
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

    void render_3d() {
        pix.fill(0x42000000);
        for (const Point& point : points) {
            render_point(point);
        }
        for (const Line& line : lines) {
            render_line(line);
        }
        for (const Surface& surface : surfaces) {
            render_surface(surface);
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

    glm::vec2 intersectionPoint(const glm::vec3 &particle_velocity,
                            const Surface &surface) {
        // Calculate the matrix A
        glm::mat3 A(surface.left_relative, surface.up_relative, particle_velocity);

        // Calculate the matrix B
        glm::vec3 B = camera_pos - surface.center;

        // Calculate the determinant of A
        float detA = glm::determinant(A);

        // Calculate the determinant of Aa
        glm::mat3 Aa(B, surface.up_relative, particle_velocity);
        float detAa = glm::determinant(Aa);

        // Calculate the determinant of Ab
        glm::mat3 Ab(surface.left_relative, B, particle_velocity);
        float detAb = glm::determinant(Ab);

        // Calculate the values of a and b
        float a = detAa / detA;
        float b = detAb / detA;

        return glm::vec2((a+1)/2, (b+1)/2);
    }

    glm::vec2 intersectisonPoint(const glm::vec3 &particle_velocity, 
                                const Surface &surface) {
        // Compute the normal of the plane using cross product
        glm::vec3 normal = glm::cross(surface.left_relative, surface.up_relative);

        // Find t for which the line intersects the plane
        float t = (glm::dot(normal, surface.center) - glm::dot(normal, camera_pos)) / glm::dot(normal, particle_velocity);

        glm::vec3 coordswap(camera_pos + t * particle_velocity);

        // Return the intersection point
        return glm::vec2((coordswap.x+1)/2, (coordswap.y+1)/2);
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
};
