#pragma once

#include <vector>

#include "../Common/ThreeDimensionScene.cpp"
#include "../../Host_Device_Shared/ManifoldData.h"

extern "C" void cuda_render_manifold(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData* manifolds, const int num_manifolds,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float opacity,
    const float ab_dilation, const float dot_opacity
);

class ManifoldScene : public ThreeDimensionScene {
private:
    int num_manifolds = 1;

public:
    ManifoldScene(const double width = 1, const double height = 1) : ThreeDimensionScene(width, height) {
        state_manager.set({
            {"manifold0_x", "(u)"},
            {"manifold0_y", "(u) 5 * sin (v) 5 * sin + 5 /"},
            {"manifold0_z", "(v)"},
            {"manifold0_r", "(u)"},
            {"manifold0_i", "(v)"},
            {"manifold0_u_min", "-3.14"},
            {"manifold0_u_max", "3.14"},
            {"manifold0_u_steps", "3000"},
            {"manifold0_v_min", "-3.14"},
            {"manifold0_v_max", "3.14"},
            {"manifold0_v_steps", "3000"},
            {"ab_dilation", ".8"},
            {"dot_opacity", "1"}
        });
    }

    void add_manifold(const string& x_eq, const string& y_eq, const string& z_eq, const string& r_eq, const string& i_eq,
                      const string& u_min, const string& u_max, const string& u_steps,
                      const string& v_min, const string& v_max, const string& v_steps) {
        const string tag = "manifold" + to_string(num_manifolds) + "_";
        state_manager.set({
            {tag + "x", x_eq},
            {tag + "y", y_eq},
            {tag + "z", z_eq},
            {tag + "r", r_eq},
            {tag + "i", i_eq},
            {tag + "u_min", u_min},
            {tag + "u_max", u_max},
            {tag + "u_steps", u_steps},
            {tag + "v_min", v_min},
            {tag + "v_max", v_max},
            {tag + "v_steps", v_steps},
        });
        num_manifolds++;
    }

    void draw() override {
        ThreeDimensionScene::draw();
        ManifoldData* manifolds = new ManifoldData[num_manifolds];
        std::vector<std::string> eqs;
        eqs.reserve(num_manifolds * 5);
        for(int i = 0; i < num_manifolds; i++) {
            const string tag = "manifold" + to_string(i) + "_";
            string x_eq = state_manager.get_equation_with_tags(tag + "x");
            string y_eq = state_manager.get_equation_with_tags(tag + "y");
            string z_eq = state_manager.get_equation_with_tags(tag + "z");
            string r_eq = state_manager.get_equation_with_tags(tag + "r");
            string i_eq = state_manager.get_equation_with_tags(tag + "i");
            int base = i * 5;
            eqs.push_back(x_eq);
            eqs.push_back(y_eq);
            eqs.push_back(z_eq);
            eqs.push_back(r_eq);
            eqs.push_back(i_eq);
            const char* x_char = eqs[base + 0].c_str();
            const char* y_char = eqs[base + 1].c_str();
            const char* z_char = eqs[base + 2].c_str();
            const char* r_char = eqs[base + 3].c_str();
            const char* i_char = eqs[base + 4].c_str();
            ManifoldData manifold{
                x_char, y_char, z_char, r_char, i_char,
                (float)state[tag + "u_min"],
                (float)state[tag + "u_max"],
                (int)state[tag + "u_steps"],
                (float)state[tag + "v_min"],
                (float)state[tag + "v_max"],
                (int)state[tag + "v_steps"]
            };
            manifolds[i] = manifold;
        }

        cuda_render_manifold(
            pix.pixels.data(),
            pix.w,
            pix.h,
            manifolds,
            num_manifolds,
            camera_pos,
            camera_direction,
            conjugate_camera_direction,
            get_geom_mean_size(),
            state["fov"],
            1, // opacity
            state["ab_dilation"],
            state["dot_opacity"]
        );
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = ThreeDimensionScene::populate_state_query();
        for(int i = 0; i < num_manifolds; i++) {
            const string tag = "manifold" + to_string(i) + "_";
            state_query_insert_multiple(s, {
                tag + "x",
                tag + "y",
                tag + "z",
                tag + "r",
                tag + "i",
                tag + "u_min",
                tag + "u_max",
                tag + "u_steps",
                tag + "v_min",
                tag + "v_max",
                tag + "v_steps"
            });
        }
        state_query_insert_multiple(s, {"ab_dilation", "dot_opacity"});
        return s;
    }
};
