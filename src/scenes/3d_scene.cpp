#pragma once

#include "scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>

class ThreeDimensionScene : public Scene {
public:
    ThreeDimensionScene(const int width, const int height) : Scene(width, height) {}
    ThreeDimensionScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    std::pair<int, int> coordinate_to_pixel(glm::vec3 coordinate) {
        // Rotate the coordinate based on the camera's orientation
        coordinate = camera_direction * (coordinate - camera_pos) * glm::conjugate(camera_direction);
        if(coordinate.z >= 0) return {-1000, -1000};

        float scale = 200.0f / coordinate.z; // perspective projection
        return {scale*coordinate.x+w/2, scale*coordinate.y+h/2};
    }

    void set_camera_position_and_orientation(glm::vec3 position, glm::quat orientation) {
        camera_pos = position;
        camera_direction = glm::normalize(orientation);
        rendered = false;
    }

    void update_variables(const std::unordered_map<string, double>& variables) {
        set_camera_position_and_orientation(
            glm::vec3(variables.at("x"), variables.at("y"), variables.at("z")),
            glm::quat(variables.at("q1"), variables.at("q2"), variables.at("q3"), variables.at("q4"))
        );
        rendered = false;
    }

    void render_3d() {
        pix.fill(BLACK);
        for (const glm::vec3& point : points) {
            render_point(point);
        }
    }

    Pixels* query(bool& done_scene) override {
        if (!rendered) {
            render_3d();
            rendered = true;
        }
        done_scene = time++ >= scene_duration_frames;
        return &pix;
    }

    void render_point(glm::vec3 point) {
        std::pair<int, int> pixel = coordinate_to_pixel(point);
        pix.fill_ellipse(pixel.first, pixel.second, 2, 2, WHITE);
    }

    void add_point(glm::vec3 point) {
        points.push_back(point);
    }

private:
    std::vector<glm::vec3> points;
    glm::vec3 camera_pos;
    glm::quat camera_direction;  // Quaternion representing the camera's orientation
};
