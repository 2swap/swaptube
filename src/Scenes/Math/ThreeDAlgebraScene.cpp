#include "ThreeDAlgebraScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

using std::complex;

extern "C" void three_d_render(
    const ivec2& wh,
    
    const quat& camera_orientation, 
    const vec3& camera_position,
    float fov_rad, 
    float max_dist,

    const float brightness,

    unsigned int* d_colors
);

ThreeDAlgebraScene::ThreeDAlgebraScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"scale", "1.0"},
        {"brightness", "1.0"},
        {"rotater", "0"},
        
// Raymarching Stuff
        {"pov_xz", "0"},
        {"pov_y", "0"},
        {"pov_fov", "3"},
        {"pov_max_dist", "10"}
    });
}

void ThreeDAlgebraScene::draw() {

    const quat camera_direction_0 = normalize(quat(cos(state["pov_xz"]), 0, sin(state["pov_xz"]), 0));
    const quat camera_direction = camera_direction_0*normalize(quat(cos(state["pov_y"]), sin(state["pov_y"])*sin(state["pov_xz"]), 0, sin(state["pov_y"])*cos(state["pov_xz"])));

    const vec3 camera_pos = rotate_vector(vec3(0,0,-state["pov_max_dist"]*0.5*state["scale"]), camera_direction);
    



    three_d_render(get_width_height(),

        camera_direction, 
        camera_pos,
        state["pov_fov"], 
        state["pov_max_dist"],
    

        state["brightness"], 
        gpu_pix.get_ptr()
    );

}
