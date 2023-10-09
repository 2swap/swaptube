#pragma once

#include "scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>

glm::quat PITCH_DOWN (0, 1 , 0, 0 );
glm::quat PITCH_UP   (0, -1, 0, 0 );
glm::quat   YAW_RIGHT(0, 0, -1, 0 );
glm::quat   YAW_LEFT (0, 0, 1 , 0 );
glm::quat  ROLL_CW   (0, 0 , 0, -1);
glm::quat  ROLL_CCW  (0, 0 , 0, 1 );

class ThreeDimensionScene : public Scene {
public:
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

    ThreeDimensionScene(const int width, const int height) : Scene(width, height) {init();}
    ThreeDimensionScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {init();}

    void init(){
        camera_direction = glm::quat(1,0,0,0);
    }

    std::pair<int, int> coordinate_to_pixel(glm::vec3 coordinate) {
        // Rotate the coordinate based on the camera's orientation
        coordinate = camera_direction * (coordinate - camera_pos) * glm::conjugate(camera_direction);
        if(coordinate.z >= 0) return {-1000, -1000};

        double scale = w*.6 / coordinate.z; // perspective projection
        return {scale*coordinate.x+w/2, scale*coordinate.y+h/2};
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
        pix.fill(BLACK);
        //for (const Point& point : points) {
        //    render_point(point);
        //}
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

    void add_point(Point point) {
        points.push_back(point);
    }

    glm::quat get_quat() {
        return camera_direction;
    }

protected:
    std::vector<Point> points;
    std::vector<Line> lines;
    glm::vec3 camera_pos;
    glm::quat camera_direction;  // Quaternion representing the camera's orientation
};
