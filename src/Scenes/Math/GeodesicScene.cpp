#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "../../Host_Device_Shared/ManifoldData.c"

ResolvedStateEquation r_eq = {
    ResolvedStateEquationComponent('a'),
    ResolvedStateEquationComponent(10.0),
    ResolvedStateEquationComponent(OP_MUL),
    ResolvedStateEquationComponent(OP_SIN),
    ResolvedStateEquationComponent('b'),
    ResolvedStateEquationComponent(10.0),
    ResolvedStateEquationComponent(OP_MUL),
    ResolvedStateEquationComponent(OP_SIN),
    ResolvedStateEquationComponent(OP_MUL),
    ResolvedStateEquationComponent(OP_ABS),
    ResolvedStateEquationComponent(0.2),
    ResolvedStateEquationComponent(OP_GT),
    ResolvedStateEquationComponent(20.0),
    ResolvedStateEquationComponent(OP_MUL),
};
ResolvedStateEquation i_eq = {
    ResolvedStateEquationComponent(0.0),
};
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
    const float ab_dilation, const float dot_radius);

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

extern "C" void cuda_render_geodesics_2d(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData& manifold,
    const glm::vec2 start_position, const glm::vec2 start_velocity,
    const int num_geodesics, const int num_steps, const float spread_angle,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov);

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
            {"manifold_d", "15.0"},
            {"manifold_q1", "1.0"},
            {"manifold_qi", "-.30"},
            {"manifold_qj", "0"},
            {"manifold_qk", "{t} .4 * sin .1 *"},
            {"manifold_fov", "1"},
            {"manifold_x", "(a)"},
            {"manifold_y", "(b)"},
            {"manifold_z", "0"},
            {"u_min", "-5.0"},
            {"u_max", "5.0"},
            {"u_steps", "1500"},
            {"v_min", "-5.0"},
            {"v_max", "5.0"},
            {"v_steps", "1500"},
            {"manifold_opacity", "1"},

            {"num_geodesics", "1"},
            {"geodesic_steps", "0"},
            {"spread_angle", "pi 2 /"},
            {"geodesics_start_u", "0.0"},
            {"geodesics_start_v", "0.0"},
            {"geodesics_start_du", "1.0"},
            {"geodesics_start_dv", "0.0"},
        });
    }

    void draw_manifold() {
        float steps_mult = geom_mean(pix.w, pix.h) / 1500.0f;
        ResolvedStateEquation x_eq = manager.get_resolved_equation("manifold_x");
        ResolvedStateEquation y_eq = manager.get_resolved_equation("manifold_y");
        ResolvedStateEquation z_eq = manager.get_resolved_equation("manifold_z");
        ManifoldData manifold1{
            x_eq.size(),
            x_eq.data(),
            y_eq.size(),
            y_eq.data(),
            z_eq.size(),
            z_eq.data(),
            r_eq.size(),
            r_eq.data(),
            i_eq.size(),
            i_eq.data(),
            (float)state["u_min"],
            (float)state["u_max"],
            (int)(state["u_steps"] * steps_mult),
            (float)state["v_min"],
            (float)state["v_max"],
            (int)(state["v_steps"] * steps_mult)
        };
        glm::quat manifold_rotation = glm::normalize(glm::quat(state["manifold_q1"], state["manifold_qi"], state["manifold_qj"], state["manifold_qk"]));
        glm::quat conjugate_manifold_rotation = glm::conjugate(manifold_rotation);
        glm::vec3 manifold_position = conjugate_manifold_rotation * glm::vec3(0.0f, 0.0f, (float)-state["manifold_d"]) * manifold_rotation;

        if(state["manifold_opacity"] >= 0.01f) {
            Pixels manifold_pix(pix.w, pix.h);
            ManifoldData manifolds[] = { manifold1 };
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
                1
            );
            cuda_overlay(
                pix.pixels.data(), pix.w, pix.h,
                manifold_pix.pixels.data(), manifold_pix.w, manifold_pix.h,
                0, 0,
                state["manifold_opacity"]
            );
        }

        int num_geodesics = (int)state["num_geodesics"];
        int geodesic_steps = (int)state["geodesic_steps"];
        if(num_geodesics > 0 && geodesic_steps > 0) {
            Pixels geodesic_pix(pix.w, pix.h);
            glm::vec2 start_position = glm::vec2(state["geodesics_start_u"], state["geodesics_start_v"]);
            glm::vec2 start_velocity = glm::vec2(state["geodesics_start_du"], state["geodesics_start_dv"]);
            cuda_render_geodesics_2d(
                geodesic_pix.pixels.data(),
                geodesic_pix.w, geodesic_pix.h,
                manifold1,
                start_position, start_velocity,
                num_geodesics, geodesic_steps,
                state["spread_angle"],
                manifold_position,
                manifold_rotation,
                conjugate_manifold_rotation,
                geom_mean(geodesic_pix.w, geodesic_pix.h),
                state["manifold_fov"]
            );
            cuda_overlay(
                pix.pixels.data(), pix.w, pix.h,
                geodesic_pix.pixels.data(), geodesic_pix.w, geodesic_pix.h,
                0, 0,
                1.0f
            );
        }
    }

    void draw() override {
        glm::vec3 camera_pos = glm::vec3(state["x"], state["y"], state["z"]);
        glm::quat camera_direction = glm::normalize(glm::quat(state["q1"], state["qi"], state["qj"], state["qk"]));
        glm::quat conjugate_camera_direction = glm::conjugate(camera_direction);

        vector<float> intensities{
            static_cast<float>(state["intensity_flat"]),
            static_cast<float>(state["intensity_sin"]),
            static_cast<float>(state["intensity_parabola"]),
            static_cast<float>(state["intensity_blackhole"]),
            static_cast<float>(state["intensity_witch"]),
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

            "num_geodesics", "geodesic_steps", "spread_angle",
            "geodesics_start_u", "geodesics_start_v", "geodesics_start_du", "geodesics_start_dv",
        };
        return sq;
    }

    bool check_if_data_changed() const { return false; }
    void change_data() {}
    void mark_data_unchanged() {}
};
