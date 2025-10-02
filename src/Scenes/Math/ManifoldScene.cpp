#pragma once

#include "../Common/ThreeDimensionScene.cpp"

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
            {"u_segs", "100"},
            {"v_min", "-3.14"},
            {"v_max", "3.14"},
            {"v_segs", "100"},
        });
    }

    void draw() override {
        ThreeDimensionScene::draw();
        cuda_render_manifold(
            pix.pixels.data(),
            pix.w,
            pix.h,
            state_manager.get_equation("manifold_x"),
            state_manager.get_equation("manifold_y"),
            state_manager.get_equation("manifold_z"),
            state_manager.get_equation("color_r"),
            state_manager.get_equation("color_i"),
            state_manager.get_value("u_min"),
            state_manager.get_value("u_max"),
            (int)state_manager.get_value("u_segs"),
            state_manager.get_value("v_min"),
            state_manager.get_value("v_max"),
            (int)state_manager.get_value("v_segs"),
            camera_pos,
            camera_direction,
            conjugate_camera_direction,
            over_w_fov,
            1, // opacity
        );
    }
};
