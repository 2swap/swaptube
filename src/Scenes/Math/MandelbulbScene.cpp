#include "MandelbulbScene.h"

extern "C" void render_raymarch(
    const int width, const int height,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iterations, const int max_mandelbulb_iterations,
    unsigned int* colors
);

MandelbulbScene::MandelbulbScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "2"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"fov", "1.5"}, 
        {"light_x", "2"}, 
        {"light_y", "4"}, 
        {"light_z", "-2"}, 
        {"max_raymarch_iterations", "127"},
        {"max_mandelbulb_iterations", "5"}
    });
};

const StateQuery MandelbulbScene::populate_state_query() const {
    return {"x", "y", "z", "d", "q1", "qi", "qj", "qk", "fov", "light_x", "light_y", "light_z", "max_mandelbulb_iterations", "max_raymarch_iterations"};
}

bool MandelbulbScene::check_if_data_changed() const {return false;}
void MandelbulbScene::change_data(){}
void MandelbulbScene::mark_data_unchanged(){}

void MandelbulbScene::draw(){
    const vec3 focus_position(state["x"], state["y"], state["z"]);
    const quat camera_quat = normalize(quat(state["q1"], state["qi"], state["qj"], state["qk"]));
    const vec3 camera_pos = focus_position + rotate_vector(vec3(0,0,-state["d"]), camera_quat);
    render_raymarch(pix.w, pix.h, 
        camera_pos, camera_quat,
        state["fov"], 
        vec3(state["light_x"], state["light_y"], state["light_z"]), 
        state["max_raymarch_iterations"], state["max_mandelbulb_iterations"],
        pix.pixels.data());
}
