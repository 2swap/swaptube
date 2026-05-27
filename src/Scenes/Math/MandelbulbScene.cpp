#include "MandelbulbScene.h"

extern "C" void render_raymarch(
    const ivec2& wh,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iterations, const int max_mandelbulb_iterations,
    uint32_t* colors
);

MandelbulbScene::MandelbulbScene(const vec2& dimensions) : Scene(dimensions), d_pixels(get_pixels_size()) {
    manager.set({
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "3"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"fov", "1.5"}, 
        {"light_x", "2"}, 
        {"light_y", "4"}, 
        {"light_z", "-2"}, 
        {"max_raymarch_iterations", "127"},
        {"max_mandelbulb_iterations", "10"}
    });
};

const StateQuery MandelbulbScene::populate_state_query() const {
    return {"x", "y", "z", "d", "q1", "qi", "qj", "qk", "fov", "light_x", "light_y", "light_z", "max_mandelbulb_iterations", "max_raymarch_iterations"};
}

void MandelbulbScene::draw(){
    const vec3 focus_position(state["x"], state["y"], state["z"]);
    const quat camera_quat = normalize(quat(state["q1"], state["qi"], state["qj"], state["qk"]));
    const vec3 camera_pos = focus_position + rotate_vector(vec3(0,0,-state["d"]), camera_quat);
    render_raymarch(pix.wh,
        camera_pos, camera_quat,
        state["fov"], 
        vec3(state["light_x"], state["light_y"], state["light_z"]), 
        state["max_raymarch_iterations"], state["max_mandelbulb_iterations"],
        d_pixels.get_ptr()
    );
    d_pixels.copy_to_host(pix.pixels.data(), pix.wh);
}
