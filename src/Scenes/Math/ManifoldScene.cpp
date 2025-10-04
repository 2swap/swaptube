#pragma once

#include "../Common/ThreeDimensionScene.cpp"

extern "C" void cuda_render_manifold(
    uint32_t* pixels, int w, int h,
    const char* manifold_x_eq, const char* manifold_y_eq, const char* manifold_z_eq,
    const char* color_r_eq, const char* color_i_eq,
    float u_min, float u_max, int u_steps,
    float v_min, float v_max, int v_steps,
    glm::vec3 camera_pos, glm::quat camera_direction, glm::quat conjugate_camera_direction,
    float geom_mean_size, float fov,
    float opacity
);

class ManifoldScene : public ThreeDimensionScene {
public:
    ManifoldScene(const double width = 1, const double height = 1) : ThreeDimensionScene(width, height) {
        state_manager.set({
            {"manifold_x", "(u)"},
            {"manifold_y", "(v)"},
            {"manifold_z", "(u) (v) + sin"},
            {"color_r", "(u)"},
            {"color_i", "(v)"},
            {"u_min", "-3.14"},
            {"u_max", "3.14"},
            {"u_segs", "2000"},
            {"v_min", "-3.14"},
            {"v_max", "3.14"},
            {"v_segs", "2000"},
        });
    }

    void draw() override {
        ThreeDimensionScene::draw();
        cuda_render_manifold(
            pix.pixels.data(),
            pix.w,
            pix.h,
            state_manager.get_equation("manifold_x").c_str(),
            state_manager.get_equation("manifold_y").c_str(),
            state_manager.get_equation("manifold_z").c_str(),
            state_manager.get_equation("color_r").c_str(),
            state_manager.get_equation("color_i").c_str(),
            state_manager.get_value("u_min"),
            state_manager.get_value("u_max"),
            (int)state_manager.get_value("u_segs"),
            state_manager.get_value("v_min"),
            state_manager.get_value("v_max"),
            (int)state_manager.get_value("v_segs"),
            camera_pos,
            camera_direction,
            conjugate_camera_direction,
            get_geom_mean_size(),
            state["fov"],
            1 // opacity
        );
        printf(state_manager.get_equation("manifold_x").c_str());
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = ThreeDimensionScene::populate_state_query();
        state_query_insert_multiple(s, {"manifold_x", "manifold_y", "manifold_z", "color_r", "color_i", "u_min", "u_max", "u_segs", "v_min", "v_max", "v_segs", "fov"});
        return s;
    }
};
