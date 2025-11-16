#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "../../Host_Device_Shared/ManifoldData.h"

extern "C" void launch_cuda_surface_raymarch(uint32_t* h_pixels, int w, int h,
    glm::quat camera_orientation, glm::vec3 camera_position,
    float fov_rad, float* intensities, float floor_distort,
    float step_size, int step_count,
    float floor_y, float ceiling_y, float grid_opacity, float zaxis);

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
        manager.set(unordered_map<string, string>{
// Raymarching Stuff
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"q1", "1"},
            {"qi", "0"},
            {"qj", "0"},
            {"qk", "0"},
            {"fov", "1"},
            {"floor_distort", "0.0"},
            {"step_size", "0.005"},
            {"step_count", "1000"},
            {"intensity_flat", "1.0 <intensity_sin> - <intensity_parabola> - <intensity_blackhole> - <intensity_witch> -"},
            {"intensity_sin", "0.0"},
            {"intensity_parabola", "0.0"},
            {"intensity_blackhole", "0.0"},
            {"intensity_witch", "0.0"},
            {"grid_opacity", "1.0"},
            {"zaxis", "0.0"},

            {"floor_y", "-1"},
            {"ceiling_y", "3"},

//Manifold Stuff
            {"manifold_d", "20.0"},
            {"manifold_q1", "1.0"},
            {"manifold_qi", "-.30"},
            {"manifold_qj", "0"},
            {"manifold_qk", "{t} .4 * sin .1 *"},
            {"manifold_fov", "1"},
            {"manifold_x", "(u)"},
            {"manifold_y", "(v)"},
            {"manifold_z", "0"},
            {"u_min", "-5.0"},
            {"u_max", "5.0"},
            {"u_steps", "3000"},
            {"v_min", "-5.0"},
            {"v_max", "5.0"},
            {"v_steps", "3000"},
            {"manifold_opacity", "1"},
        });
    }

    void draw_manifold() {
        if(state["manifold_opacity"] <= 0.01f) return;
        Pixels manifold_pix(pix.w, pix.h);
        float steps_mult = geom_mean(manifold_pix.w, manifold_pix.h) / 1500.0f;
        string x_eq = manager.get_equation_with_tags("manifold_x");
        string y_eq = manager.get_equation_with_tags("manifold_y");
        string z_eq = manager.get_equation_with_tags("manifold_z");
        ManifoldData manifold1{
            x_eq.c_str(),
            y_eq.c_str(),
            z_eq.c_str(),
            "(u) 10 * sin (v) 10 * sin * abs .2 > 20 *", "0",
            (float)state["u_min"],
            (float)state["u_max"],
            (int)(state["u_steps"] * steps_mult),
            (float)state["v_min"],
            (float)state["v_max"],
            (int)(state["v_steps"] * steps_mult)
        };
        ManifoldData manifolds[] = { manifold1 };

        glm::quat manifold_rotation = glm::normalize(glm::quat(state["manifold_q1"], state["manifold_qi"], state["manifold_qj"], state["manifold_qk"]));
        glm::quat conjugate_manifold_rotation = glm::conjugate(manifold_rotation);
        glm::vec3 manifold_position = conjugate_manifold_rotation * glm::vec3(0.0f, 0.0f, (float)-state["manifold_d"]) * manifold_rotation;
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
            state["manifold_fov"],
            1,
            1,
            0
        );

        cuda_overlay(
            pix.pixels.data(), pix.w, pix.h,
            manifold_pix.pixels.data(), manifold_pix.w, manifold_pix.h,
            0, 0,
            state["manifold_opacity"]
        );
    }

    void draw() override {
        glm::vec3 camera_pos = glm::vec3(state["x"], state["y"], state["z"]);
        glm::quat camera_direction = glm::normalize(glm::quat(state["q1"], state["qi"], state["qj"], state["qk"]));
        glm::quat conjugate_camera_direction = glm::conjugate(camera_direction);

        vector<float> intensities{
            (float)state["intensity_flat"],
            (float)state["intensity_sin"],
            (float)state["intensity_parabola"],
            (float)state["intensity_blackhole"],
            (float)state["intensity_witch"]
        };
        launch_cuda_surface_raymarch(pix.pixels.data(), get_width(), get_height(),
                                     camera_direction, camera_pos,
                                     state["fov"], intensities.data(), state["floor_distort"],
                                     state["step_size"], (int)state["step_count"],
                                     state["floor_y"], state["ceiling_y"], state["grid_opacity"], state["zaxis"]);

        draw_manifold();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = {
            "x", "y", "z", "q1", "qi", "qj", "qk",
            "fov", "floor_distort",
            "step_size", "step_count",
            "grid_opacity", "zaxis",
            "floor_y", "ceiling_y",
            "intensity_flat", "intensity_sin", "intensity_parabola", "intensity_blackhole", "intensity_witch",

            "manifold_d", "manifold_q1", "manifold_qi", "manifold_qj", "manifold_qk", "manifold_fov",
            "u_min", "u_max", "u_steps", "v_min", "v_max", "v_steps",
            "manifold_opacity",
        };
        return sq;
    }

    bool check_if_data_changed() const { return false; }
    void change_data() {}
    void mark_data_unchanged() {}
};
