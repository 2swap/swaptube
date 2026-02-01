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

extern "C" void cuda_render_geodesics_2d(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData& manifold,
    const glm::vec2 start_position, const glm::vec2 start_velocity,
    const int num_geodesics, const int num_steps, const float spread_angle,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov, const float opacity);

class GeodesicScene : public Scene {
public:
    GeodesicScene(const double width = 1, const double height = 1)
        : Scene(width, height) {
        manager.set(unordered_map<string, string>{
// Raymarching Stuff
            {"pov_x", "0"},
            {"pov_y", "0"},
            {"pov_z", "0"},
            {"pov_q1", "1"},
            {"pov_qi", "0"},
            {"pov_qj", "0"},
            {"pov_qk", "0"},
            {"pov_fov", "1"},
            {"pov_floor_distort", "0.0"},
            {"pov_step_size", "0.05"},
            {"pov_step_count", "1000"},
            {"pov_intensity_flat", "1.0 <pov_intensity_sin> - <pov_intensity_parabola> - <pov_intensity_blackhole> - <pov_intensity_witch> -"},
            {"pov_intensity_sin", "0.0"},
            {"pov_intensity_parabola", "0.0"},
            {"pov_intensity_blackhole", "0.0"},
            {"pov_intensity_witch", "0.0"},
            {"pov_grid_opacity", "1.0"},
            {"pov_zaxis", "0.0"},

            {"pov_floor_y", "-1"},
            {"pov_ceiling_y", "3"},

//Manifold Stuff
            {"manifold_d", "15.0"},
            {"manifold_q1", "1.0"},
            {"manifold_qi", "-.2"},
            {"manifold_qj", "0"},
            {"manifold_qk", "{t} .4 * sin .1 *"},
            {"manifold_fov", "1"},
            {"manifold_x", "(a)"},
            {"manifold_y", "(b)"},
            {"manifold_z", "0"},
            {"manifold_u_min", "-5.0"},
            {"manifold_u_max", "5.0"},
            {"manifold_u_steps", "1500"},
            {"manifold_v_min", "-5.0"},
            {"manifold_v_max", "5.0"},
            {"manifold_v_steps", "1500"},
            {"manifold_opacity", "1"},

            {"geodesics_count", "1"},
            {"geodesics_steps", "0"},
            {"geodesics_spread_angle", "pi 2 /"},
            {"geodesics_start_u", "0.0"},
            {"geodesics_start_v", "0.0"},
            {"geodesics_start_du", "1.0"},
            {"geodesics_start_dv", "0.0"},
            {"geodesics_opacity", "1.0"},
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
            (float)state["manifold_u_min"],
            (float)state["manifold_u_max"],
            (int)(state["manifold_u_steps"] * steps_mult),
            (float)state["manifold_v_min"],
            (float)state["manifold_v_max"],
            (int)(state["manifold_v_steps"] * steps_mult),
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

        int num_geodesics = (int)state["geodesics_count"];
        int geodesic_steps = (int)state["geodesics_steps"];
        double geodesics_opacity = state["geodesics_opacity"];
        if(num_geodesics > 0 && geodesic_steps > 0 && geodesics_opacity >= 0.01f) {
            Pixels geodesic_pix(pix.w, pix.h);
            glm::vec2 start_position = glm::vec2(state["geodesics_start_u"], state["geodesics_start_v"]);
            glm::vec2 start_velocity = glm::vec2(state["geodesics_start_du"], state["geodesics_start_dv"]);
            cuda_render_geodesics_2d(
                geodesic_pix.pixels.data(),
                geodesic_pix.w, geodesic_pix.h,
                manifold1,
                start_position, start_velocity,
                num_geodesics, geodesic_steps,
                state["geodesics_spread_angle"],
                manifold_position,
                manifold_rotation,
                conjugate_manifold_rotation,
                geom_mean(geodesic_pix.w, geodesic_pix.h),
                state["manifold_fov"],
                state["geodesics_opacity"]
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
        glm::vec3 camera_pos = glm::vec3(state["pov_x"], state["pov_y"], state["pov_z"]);
        glm::quat camera_direction = glm::normalize(glm::quat(state["pov_q1"], state["pov_qi"], state["pov_qj"], state["pov_qk"]));
        glm::quat conjugate_camera_direction = glm::conjugate(camera_direction);

        vector<float> intensities{
            static_cast<float>(state["pov_intensity_flat"]),
            static_cast<float>(state["pov_intensity_sin"]),
            static_cast<float>(state["pov_intensity_parabola"]),
            static_cast<float>(state["pov_intensity_blackhole"]),
            static_cast<float>(state["pov_intensity_witch"]),
        };
        launch_cuda_surface_raymarch(pix.pixels.data(), get_width(), get_height(),
                                     camera_direction, camera_pos,
                                     state["pov_fov"], intensities.data(), state["pov_floor_distort"],
                                     state["pov_step_size"], (int)state["pov_step_count"],
                                     state["pov_floor_y"], state["pov_ceiling_y"], state["pov_grid_opacity"], state["pov_zaxis"]);

        draw_manifold();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = {
            "pov_x", "pov_y", "pov_z",
            "pov_q1", "pov_qi", "pov_qj", "pov_qk",
            "pov_fov",
            "pov_step_size", "pov_step_count",
            "pov_grid_opacity",
            "pov_zaxis",
            "pov_floor_y", "pov_ceiling_y", "pov_floor_distort",
            "pov_intensity_flat", "pov_intensity_sin", "pov_intensity_parabola", "pov_intensity_blackhole", "pov_intensity_witch",

            "manifold_d", "manifold_q1", "manifold_qi", "manifold_qj", "manifold_qk", "manifold_fov",
            "manifold_u_min", "manifold_u_max", "manifold_u_steps", "manifold_v_min", "manifold_v_max", "manifold_v_steps",
            "manifold_opacity",

            "geodesics_count", "geodesics_steps", "geodesics_spread_angle", "geodesics_opacity",
            "geodesics_start_u", "geodesics_start_v", "geodesics_start_du", "geodesics_start_dv",
        };
        return sq;
    }

    bool check_if_data_changed() const { return false; }
    void change_data() {}
    void mark_data_unchanged() {}
};
