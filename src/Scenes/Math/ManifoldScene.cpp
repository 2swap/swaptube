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
        });
    }

    void draw() override {
        ThreeDimensionScene::draw();
        draw_manifold_cuda(
            state_manager.get_equation("manifold_x"),
            state_manager.get_equation("manifold_y"),
            state_manager.get_equation("manifold_z"),
            state_manager.get_equation("color_r"),
            state_manager.get_equation("color_i"),
        );
    void cuda_render_surface(
        vector<unsigned int>& pix,
        int x1,
        int y1,
        int plot_w,
        int plot_h,
        int pixels_w,
        unsigned int* d_surface,
        int surface_w,
        int surface_h,
        float opacity,
        glm::vec3 camera_pos,
        glm::quat camera_direction,
        glm::quat conjugate_camera_direction,
        const glm::vec3& surface_normal,
        const glm::vec3& surface_center,
        const glm::vec3& surface_pos_x_dir,
        const glm::vec3& surface_pos_y_dir,
        const float surface_ilr2,
        const float surface_iur2,
        float halfwidth,
        float halfheight,
        float over_w_fov);
    }
};
