#include "VolumetricFractalScene.h"

extern "C" void render_volume(
    const int width, const int height,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    float p,
    unsigned int* colors
);

extern "C" void allocate_array(const int width, const int height, const int depth, float** d_pointer);

extern "C" void compute_voxel_array(const int width, const int height, const int depth, float* voxel_data);

extern "C" void render_voxel_array(const int image_width, const int image_height, const vec3 pos, const quat camera_orientation, const float fov, const int voxel_width, const int voxel_height, const int voxel_depth, unsigned int* colors, float* voxel_data);

extern "C" void freeData(float* data);

VolumetricScene::VolumetricScene(const vec2& dimensions) : Scene(dimensions), voxel_data(nullptr){
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
        {"max_mandelbulb_iterations", "5"},
        {"power", "2"},
        {"voxel_grid_size", "100"}
    });
};

VolumetricScene::~VolumetricScene(){
    if(voxel_data){
        freeData(voxel_data);
    }
}

const StateQuery VolumetricScene::populate_state_query() const {
    return {"x", "y", "z", "d", "q1", "qi", "qj", "qk", "fov", "light_x", "light_y", "light_z", "max_mandelbulb_iterations", "max_raymarch_iterations", "p", "voxel_grid_size"};
}

void VolumetricScene::load_voxels(const int width, const int height, const int depth){ // x, y, z dimensions
    allocate_array(width, height, depth, &voxel_data);
    compute_voxel_array(width, height, depth, voxel_data);
}

bool VolumetricScene::check_if_data_changed() const {return false;}
void VolumetricScene::change_data(){}
void VolumetricScene::mark_data_unchanged(){}

void VolumetricScene::draw(){
    const vec3 focus_position(state["x"], state["y"], state["z"]);
    const quat camera_quat = normalize(quat(state["q1"], state["qi"], state["qj"], state["qk"]));
    const vec3 camera_pos = focus_position + rotate_vector(vec3(0,0,-state["d"]), camera_quat);
    render_volume(pix.w, pix.h, 
        camera_pos, camera_quat,
        state["fov"], 
        vec3(state["light_x"], state["light_y"], state["light_z"]), 
        state["max_raymarch_iterations"], state["max_mandelbulb_iterations"],
        state["power"],
        pix.pixels.data());
}
