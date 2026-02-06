#pragma once

#include "../Scene.cpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "../../Host_Device_Shared/ManifoldData.c"

ResolvedStateEquation r_eq = {
    {RESOLVED_CUDA_TAG, .content = {.cuda_tag = 0}},
    {RESOLVED_CONSTANT, .content = {.constant = 10.0}},
    {RESOLVED_OPERATOR, .content = {.op = OP_MUL}},
    {RESOLVED_OPERATOR, .content = {.op = OP_SIN}},
    {RESOLVED_CUDA_TAG, .content = {.cuda_tag = 1}},
    {RESOLVED_CONSTANT, .content = {.constant = 10.0}},
    {RESOLVED_OPERATOR, .content = {.op = OP_MUL}},
    {RESOLVED_OPERATOR, .content = {.op = OP_SIN}},
    {RESOLVED_OPERATOR, .content = {.op = OP_MUL}},
    {RESOLVED_OPERATOR, .content = {.op = OP_ABS}},
    {RESOLVED_CONSTANT, .content = {.constant = 0.2}},
    {RESOLVED_OPERATOR, .content = {.op = OP_GT}},
    {RESOLVED_CONSTANT, .content = {.constant = 20.0}},
    {RESOLVED_OPERATOR, .content = {.op = OP_MUL}},
};
ResolvedStateEquation i_eq = {
    {RESOLVED_CONSTANT, .content = {.constant = 0.0}},
};
extern "C" void launch_cuda_surface_raymarch(
    uint32_t* h_pixels, int w, int h,
    int x_size, ResolvedStateEquationComponent* x_eq,
    int y_size, ResolvedStateEquationComponent* y_eq,
    int z_size, ResolvedStateEquationComponent* z_eq,
    int w_size, ResolvedStateEquationComponent* w_eq,
    int is_special,
    glm::quat camera_orientation, glm::vec3 camera_position,
    float fov_rad, float max_dist,
    float floor_y, float ceiling_y, float grid_thickness);

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
            {"space_x", "(a)"},
            {"space_y", "(b)"},
            {"space_z", "(c)"},
            {"space_w", "0"},

// Raymarching Stuff
            {"pov_x", "0"},
            {"pov_y", "0"},
            {"pov_z", "0"},
            {"pov_q1", "1"},
            {"pov_qi", "0"},
            {"pov_qj", "0"},
            {"pov_qk", "0"},
            {"pov_fov", "2"},
            {"pov_max_dist", "5"},
            {"pov_floor_y", "-1"},
            {"pov_ceiling_y", "1"},
            {"pov_grid_thickness", "0.1"},

//Manifold Stuff
            {"manifold_d", "15.0"},
            {"manifold_fov", "1"},
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

    void draw_perspective(ResolvedStateEquation& x_eq,
                          ResolvedStateEquation& y_eq,
                          ResolvedStateEquation& z_eq,
                          ResolvedStateEquation& w_eq) {
        glm::vec3 camera_pos = glm::vec3(state["pov_x"], state["pov_y"], state["pov_z"]);
        glm::quat camera_direction = glm::normalize(glm::quat(state["pov_q1"], state["pov_qi"], state["pov_qj"], state["pov_qk"]));
        glm::quat conjugate_camera_direction = glm::conjugate(camera_direction);

        bool x_y_z_flat = (
            x_eq.size() == 1 && x_eq.data()[0].type == RESOLVED_CUDA_TAG && x_eq.data()[0].content.cuda_tag == 0 &&
            y_eq.size() == 1 && y_eq.data()[0].type == RESOLVED_CUDA_TAG && y_eq.data()[0].content.cuda_tag == 1 &&
            z_eq.size() == 1 && z_eq.data()[0].type == RESOLVED_CUDA_TAG && z_eq.data()[0].content.cuda_tag == 2
        );
        bool w_flat = w_eq.size() == 1;
        int special_case_code = 0;
        if(x_y_z_flat) special_case_code = 1;
        if(x_y_z_flat && w_flat) special_case_code = 2;

        launch_cuda_surface_raymarch(pix.pixels.data(), get_width(), get_height(),
                                     x_eq.size(), x_eq.data(),
                                     y_eq.size(), y_eq.data(),
                                     z_eq.size(), z_eq.data(),
                                     w_eq.size(), w_eq.data(),
                                     special_case_code,
                                     camera_direction, camera_pos,
                                     state["pov_fov"], state["pov_max_dist"],
                                     state["pov_floor_y"], state["pov_ceiling_y"], state["pov_grid_thickness"]);
    }

    void draw_manifold(ResolvedStateEquation& x_eq,
                       ResolvedStateEquation& y_eq,
                       ResolvedStateEquation& z_eq,
                       ResolvedStateEquation& w_eq) {
        float steps_mult = geom_mean(pix.w, pix.h) / 1500.0f;
        ManifoldData manifold1{
            x_eq.size(),
            x_eq.data(),
            y_eq.size(),
            y_eq.data(),
            w_eq.size(),
            w_eq.data(),
            r_eq.size(),
            r_eq.data(),
            i_eq.size(),
            i_eq.data(),
            -1.0f,
            1.0f,
            (int)(1500 * steps_mult),
            -1.0f,
            1.0f,
            (int)(1500 * steps_mult),
        };
        glm::quat manifold_rotation = glm::normalize(glm::quat(state["pov_q1"], state["pov_qi"], state["pov_qj"], state["pov_qk"]));
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
            glm::vec2 start_position = glm::vec2(state["pov_x"], state["pov_z"]);
            glm::vec2 start_velocity = glm::vec2(state["pov_q1"], state["pov_qj"]);
            start_velocity = glm::normalize(start_velocity);
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
        ResolvedStateEquation x_eq = manager.get_resolved_equation("space_x");
        ResolvedStateEquation y_eq = manager.get_resolved_equation("space_y");
        ResolvedStateEquation z_eq = manager.get_resolved_equation("space_z");
        ResolvedStateEquation w_eq = manager.get_resolved_equation("space_w");

        draw_perspective(x_eq, y_eq, z_eq, w_eq);

        //draw_manifold(x_eq, y_eq, z_eq, w_eq);
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = {
            "space_x", "space_y", "space_z", "space_w",

            "pov_x", "pov_y", "pov_z",
            "pov_q1", "pov_qi", "pov_qj", "pov_qk",
            "pov_fov", "pov_max_dist",
            "pov_floor_y", "pov_ceiling_y",
            "pov_grid_thickness",

            "manifold_d", "manifold_fov",
            "manifold_opacity",

            "geodesics_count", "geodesics_steps", "geodesics_spread_angle", "geodesics_opacity",
        };
        return sq;
    }

    bool check_if_data_changed() const { return false; }
    void change_data() {}
    void mark_data_unchanged() {}
};
