#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "../../Host_Device_Shared/ManifoldData.h"

extern "C" void launch_cuda_surface_raymarch(uint32_t* h_pixels, int w, int h,
                                             glm::quat camera_orientation, glm::vec3 camera_position,
                                             float fov_rad, float intensity);

extern "C" void cuda_render_manifold(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData* manifolds, const int num_manifolds,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float ab_dilation, const float dot_radius,
    const float axes_length);

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

class GeodesicScene : public Scene {
public:
    GeodesicScene(const double width = 1, const double height = 1)
        : Scene(width, height) {
        state.set(unordered_map<string, string>{
// Raymarching Stuff
            {"fov", "1"},
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "0"},
            {"intensity", "1.0"},

//Manifold Stuff
            {"manifold_d", "50.0"},
            {"manifold_q1", "1.0"},
            {"manifold_qi", "{t} sin"},
            {"manifold_qj", "{t} cos"},
            {"manifold_qk", "0.0"},
            {"manifold_fov", "1"},
            {"u_min", "-10.0"},
            {"u_max", "10.0"},
            {"u_steps", "3000"},
            {"v_min", "-10.0"},
            {"v_max", "10.0"},
            {"v_steps", "3000"}
        });
    }

    void draw_manifold() {
        float geom_mean_size = get_geom_mean_size();
        float steps_mult = geom_mean_size / 3.0f / 1500.0f;
        ManifoldData manifold1{
            "(u)", "(v)", "1 (u) (u) * (v) (v) * + /", "(u) 10 /", "(v) 10 /",
            (float)state["u_min"],
            (float)state["u_max"],
            (int)(state["u_steps"] * steps_mult),
            (float)state["v_min"],
            (float)state["v_max"],
            (int)(state["v_steps"] * steps_mult)
        };
        ManifoldData manifolds[] = { manifold1 };

        Pixels manifold_pix(pix.w / 3, pix.h / 3);
        glm::quat manifold_rotation = glm::normalize(glm::quat(state["manifold_q1"], state["manifold_qi"], state["manifold_qj"], state["manifold_qk"]));
        glm::quat conjugate_manifold_rotation = glm::conjugate(manifold_rotation);
        glm::vec3 manifold_position = conjugate_manifold_rotation * glm::vec3(0,0,-state["manifold_d"]) * manifold_rotation;
        cuda_render_manifold(
            manifold_pix.pixels.data(),
            manifold_pix.w,
            manifold_pix.h,
            manifolds,
            1,
            manifold_position,
            manifold_rotation,
            conjugate_manifold_rotation,
            geom_mean(manifold_pix.w, manifold_pix.h),
            current_state["manifold_fov"],
            1,
            1,
            0
        );

        cuda_overlay(
            pix.pixels.data(), pix.w, pix.h,
            manifold_pix.pixels.data(), manifold_pix.w, manifold_pix.h,
            pix.w - manifold_pix.w, pix.h - manifold_pix.h,
            1
        );
    }

    void draw() override {
        glm::vec3 camera_pos = glm::vec3(state["x"], state["y"], state["z"]);
        glm::quat camera_direction = glm::normalize(glm::quat(state["q1"], state["qi"], state["qj"], state["qk"]));
        glm::quat conjugate_camera_direction = glm::conjugate(camera_direction);

        launch_cuda_surface_raymarch(pix.pixels.data(), get_width(), get_height(),
                                     camera_direction, camera_pos,
                                     state["fov"], state["intensity"]);

        draw_manifold();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = { "fov", "x", "y", "z", "q1", "qi", "qj", "qk", "intensity", "u_min", "u_max", "u_steps", "v_min", "v_max", "v_steps" };
        return sq;
    }

    bool check_if_data_changed() const { return false; }
    void change_data() {}
    void mark_data_unchanged() {}
};
