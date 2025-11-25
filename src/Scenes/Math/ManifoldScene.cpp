#pragma once

#include <vector>

#include "../Common/ThreeDimensionScene.cpp"
#include "../../Host_Device_Shared/ManifoldData.c"

extern "C" void cuda_render_manifold(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData* manifolds, const int num_manifolds,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float ab_dilation, const float dot_radius
);

class ManifoldScene : public ThreeDimensionScene {
private:
    unordered_set<string> manifold_names;

public:
    ManifoldScene(const double width = 1, const double height = 1) : ThreeDimensionScene(width, height) {
        manager.set({
            {"ab_dilation", ".8"},
            {"dot_radius", "1"},
        });
    }

    void add_manifold(const string& name,
                      const string& x_eq, const string& y_eq, const string& z_eq, const string& r_eq, const string& i_eq,
                      const string& u_min, const string& u_max, const string& u_steps,
                      const string& v_min, const string& v_max, const string& v_steps) {
        const string tag = "manifold" + name + "_";
        manager.set({
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
        manager.remove({
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
        set_camera_direction();

        ManifoldData* manifolds = new ManifoldData[manifold_names.size()];
        int i = 0;
        float geom_mean_size = get_geom_mean_size();
        float steps_mult = geom_mean_size / 1500.0f;
        vector<ResolvedStateEquation> resolved_equations;
        for(const string& name : manifold_names) {
            const string tag = "manifold" + name + "_";
            resolved_equations.push_back(manager.get_resolved_equation(tag + "x"));
            resolved_equations.push_back(manager.get_resolved_equation(tag + "y"));
            resolved_equations.push_back(manager.get_resolved_equation(tag + "z"));
            resolved_equations.push_back(manager.get_resolved_equation(tag + "r"));
            resolved_equations.push_back(manager.get_resolved_equation(tag + "i"));
            ResolvedStateEquation& x_eq = resolved_equations[resolved_equations.size() - 5];
            ResolvedStateEquation& y_eq = resolved_equations[resolved_equations.size() - 4];
            ResolvedStateEquation& z_eq = resolved_equations[resolved_equations.size() - 3];
            ResolvedStateEquation& r_eq = resolved_equations[resolved_equations.size() - 2];
            ResolvedStateEquation& i_eq = resolved_equations[resolved_equations.size() - 1];
            ManifoldData manifold{
                x_eq.size(), x_eq.data(),
                y_eq.size(), y_eq.data(),
                z_eq.size(), z_eq.data(),
                r_eq.size(), r_eq.data(),
                i_eq.size(), i_eq.data(),
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
            state["dot_radius"]
        );
        ThreeDimensionScene::draw();
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
        state_query_insert_multiple(s, {"ab_dilation", "dot_radius"});
        return s;
    }
};
