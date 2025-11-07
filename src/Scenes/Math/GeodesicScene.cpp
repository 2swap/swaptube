#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

extern "C" void launch_cuda_surface_raymarch(uint32_t* h_pixels, int w, int h,
                                             glm::quat camera_orientation, glm::vec3 camera_position,
                                             float fov_rad, const char* w_eq);

class GeodesicScene : public Scene {
public:
    GeodesicScene(const double width = 1, const double height = 1)
        : Scene(width, height) {
        state.set(unordered_map<string, string>{
            {"fov", "1"},
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "0"},
            {"w_eq", "0"},
        });
    }

    void draw() override {
        glm::vec3 camera_pos = glm::vec3(state["x"], state["y"], state["z"]);
        glm::quat camera_direction = glm::normalize(glm::quat(state["q1"], state["qi"], state["qj"], state["qk"]));

        string w_eq = state.get_equation_with_tags("w_eq");
        launch_cuda_surface_raymarch(pix.pixels.data(), get_width(), get_height(),
                                     camera_direction, camera_pos,
                                     state["fov"], w_eq.c_str());
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = { "fov", "x", "y", "z", "q1", "qi", "qj", "qk", "w_eq" };
        return sq;
    }

    bool check_if_data_changed() const { return false; }
    void change_data() {}
    void mark_data_unchanged() {}
};
