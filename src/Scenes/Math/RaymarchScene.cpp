#include "RaymarchScene.h"

extern "C" void render_raymarch(
    const int width, const int height,
    vec3 cameraPos, vec3 cameraDir, vec3 cameraUp, float fov,
    vec3 lightPos, int maxIters,
    unsigned int* colors
);

RaymarchScene::RaymarchScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"camera_x", "0"},
        {"camera_y", "1"}, 
        {"camera_z", "-2"}, 
        {"cameraDir_x", "<camera_x> -1 *"}, 
        {"cameraDir_y", "-1"}, 
        {"cameraDir_z", "<camera_z> -1 *"}, 
        {"cameraUp_x", "<camera_x> -0.5 *"}, 
        {"cameraUp_y", "2"}, 
        {"cameraUp_z", "<camera_z> -0.5 *"}, 
        {"fov_degrees", "90"}, 
        {"light_x", "2"}, 
        {"light_y", "4"}, 
        {"light_z", "-2"}, 
        {"max_iterations", "127"}
    });
};

const StateQuery RaymarchScene::populate_state_query() const {
    StateQuery sq = {"camera_x", "camera_y", "camera_z", "cameraDir_x", "cameraDir_y", "cameraDir_z", "cameraUp_x", "cameraUp_y", "cameraUp_z", "fov_degrees", "light_x", "light_y", "light_z", "max_iterations"};
    return sq;
}

bool RaymarchScene::check_if_data_changed() const {return false;}
void RaymarchScene::change_data(){}
void RaymarchScene::mark_data_unchanged(){}

void RaymarchScene::draw(){
    render_raymarch(pix.w, pix.h, 
        vec3(state["camera_x"], state["camera_y"], state["camera_z"]), 
        vec3(state["cameraDir_x"], state["cameraDir_y"], state["cameraDir_z"]), 
        vec3(state["cameraUp_x"], state["cameraUp_y"], state["cameraUp_z"]), 
        state["fov_degrees"], 
        vec3(state["light_x"], state["light_y"], state["light_z"]), 
        state["max_iterations"], pix.pixels.data());
}