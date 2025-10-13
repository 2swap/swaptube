#pragma once

#include <vector>

#include "../Common/ThreeDimensionScene.cpp"
#include "../../Host_Device_Shared/ManifoldData.h"

extern "C" void cuda_render_manifold(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData* manifolds, const int num_manifolds,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float ab_dilation, const float dot_radius,
    const float axes_length
);

class ManifoldScene : public ThreeDimensionScene {
private:
    unordered_set<string> manifold_names;

public:
    ManifoldScene(const double width = 1, const double height = 1) : ThreeDimensionScene(width, height) {
        state.set({
            {"ab_dilation", ".8"},
            {"dot_radius", "1"},
            {"axes_length", "1"},
        });
    }

    void add_manifold(const string& name,
                      const string& x_eq, const string& y_eq, const string& z_eq, const string& r_eq, const string& i_eq,
                      const string& u_min, const string& u_max, const string& u_steps,
                      const string& v_min, const string& v_max, const string& v_steps) {
        const string tag = "manifold" + name + "_";
        state.set({
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
        manifold_names.insert(name);
    }

    void remove_manifold(const string& name) {
        const string tag = "manifold" + name + "_";
        state.remove({
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
        manifold_names.erase(name);
    }

    void draw() override {
        ThreeDimensionScene::draw();
        ManifoldData* manifolds = new ManifoldData[manifold_names.size()];
        std::vector<std::string> eqs;
        eqs.reserve(manifold_names.size() * 5);
        int i = 0;
        float geom_mean_size = get_geom_mean_size();
        float steps_mult = geom_mean_size / 1500.0f;
        for(const string& name : manifold_names) {
            const string tag = "manifold" + name + "_";
            string x_eq = state.get_equation_with_tags(tag + "x");
            string y_eq = state.get_equation_with_tags(tag + "y");
            string z_eq = state.get_equation_with_tags(tag + "z");
            string r_eq = state.get_equation_with_tags(tag + "r");
            string i_eq = state.get_equation_with_tags(tag + "i");
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
                (int)(state[tag + "u_steps"] * steps_mult),
                (float)state[tag + "v_min"],
                (float)state[tag + "v_max"],
                (int)(state[tag + "v_steps"] * steps_mult)
            };
            manifolds[i] = manifold;
            i++;
        }

        cuda_render_manifold(
            pix.pixels.data(),
            pix.w,
            pix.h,
            manifolds,
            manifold_names.size(),
            camera_pos,
            camera_direction,
            conjugate_camera_direction,
            geom_mean_size,
            state["fov"],
            state["ab_dilation"],
            state["dot_radius"],
            state["axes_length"]
        );
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = ThreeDimensionScene::populate_state_query();
        for(const string& name : manifold_names) {
            const string tag = "manifold" + name + "_";
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
        state_query_insert_multiple(s, {"ab_dilation", "dot_radius", "axes_length"});
        return s;
    }
};
