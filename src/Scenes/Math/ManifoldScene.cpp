#include <vector>

#include "ManifoldScene.h"

extern "C" void cuda_render_manifold(
    Color* pixels, const int w, const int h,
    const ManifoldData* manifolds, const int num_manifolds,
    const vec3 camera_pos, const quat camera_direction,
    const float geom_mean_size, const float fov,
    const float ab_dilation, const float dot_radius,
    const Color* tex_pixels, const int tex_w, const int tex_h
);
extern "C" void cuda_free_texture(Color* d_tex_pixels);
extern "C" Color* cuda_copy_texture_to_device(const Color* h_tex_pixels, const int tex_w, const int tex_h);

ManifoldScene::ManifoldScene(const vec2& dimensions) : ThreeDimensionScene(dimensions), d_texture_data(nullptr), texture_w(0), texture_h(0) {
    manager.set({
        {"ab_dilation", ".8"},
        {"dot_radius", "1"},
    });
}

void ManifoldScene::add_manifold(const std::string& name,
                      const std::string& x_eq, const std::string& y_eq, const std::string& z_eq, const std::string& r_eq, const std::string& i_eq,
                      const std::string& u_min, const std::string& u_max, const std::string& u_steps,
                      const std::string& v_min, const std::string& v_max, const std::string& v_steps) {
    const std::string tag = "manifold" + name + "_";
    manager.set({
        {tag + "x", x_eq},
        {tag + "y", y_eq},
        {tag + "z", z_eq},
        {tag + "r", r_eq},
        {tag + "i", i_eq},
        {tag + "a_min", u_min},
        {tag + "a_max", u_max},
        {tag + "a_steps", u_steps},
        {tag + "b_min", v_min},
        {tag + "b_max", v_max},
        {tag + "b_steps", v_steps},
    });
    manifold_names.insert(name);
}

void ManifoldScene::remove_manifold(const std::string& name) {
    const std::string tag = "manifold" + name + "_";
    manager.remove({
        tag + "x",
        tag + "y",
        tag + "z",
        tag + "r",
        tag + "i",
        tag + "a_min",
        tag + "a_max",
        tag + "a_steps",
        tag + "b_min",
        tag + "b_max",
        tag + "b_steps"
    });
    manifold_names.erase(name);
}

void ManifoldScene::draw() {
    set_camera_direction();

    ManifoldData* manifolds = new ManifoldData[manifold_names.size()];
    int i = 0;
    float geom_mean_size = get_geom_mean_size();
    float steps_mult = geom_mean_size / 1500.0f;
    std::vector<ResolvedStateEquation> resolved_equations;
    for(const std::string& name : manifold_names) {
        const std::string tag = "manifold" + name + "_";
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
            (float)state[tag + "a_min"],
            (float)state[tag + "a_max"],
            (int)(state[tag + "a_steps"] * steps_mult),
            (float)state[tag + "b_min"],
            (float)state[tag + "b_max"],
            (int)(state[tag + "b_steps"] * steps_mult)
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
        geom_mean_size,
        state["fov"],
        state["ab_dilation"],
        state["dot_radius"],
        d_texture_data,
        texture_w,
        texture_h
    );
    ThreeDimensionScene::draw();
}

const StateQuery ManifoldScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    for(const std::string& name : manifold_names) {
        const std::string tag = "manifold" + name + "_";
        state_query_insert_multiple(s, {
            tag + "x",
            tag + "y",
            tag + "z",
            tag + "r",
            tag + "i",
            tag + "a_min",
            tag + "a_max",
            tag + "a_steps",
            tag + "b_min",
            tag + "b_max",
            tag + "b_steps"
        });
    }
    state_query_insert_multiple(s, {"ab_dilation", "dot_radius"});
    return s;
}

void ManifoldScene::set_texture(const Pixels& new_texture) {
    if(d_texture_data) {
        cuda_free_texture(d_texture_data);
    }
    d_texture_data = cuda_copy_texture_to_device(new_texture.pixels.data(), new_texture.w, new_texture.h);
    texture_w = new_texture.w;
    texture_h = new_texture.h;
}

ManifoldScene::~ManifoldScene() {
    if(d_texture_data) {
        cuda_free_texture(d_texture_data);
    }
}
