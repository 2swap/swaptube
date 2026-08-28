#include "FourDAlgebraScene.h"
#include "FourDPlaneScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

using std::complex;

extern "C" void four_d_render(
    const ivec2& wh,
    
    const quat& camera_orientation, 
    const vec3& camera_position,
    float fov_rad, 
    float max_dist,

    vec4 x_unit,
    vec4 y_unit,
    vec4 z_unit,
    vec4 rotater,
    vec4 rotaterInv,

    vec4 jj,
    vec4 ijj,

    const float brightness,
    const vec3 channels,
    const float fade,
    const float slider,
    const int equation,

    unsigned int* d_colors
);

FourDAlgebraScene::FourDAlgebraScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"rotation_1k", "0"},
        {"rotation_ik", "0"},
        {"rotation_jk", "0"},
        {"scale", "1.0"},

        {"brightness", "1.0"},
        {"r_channel", "1.0"},
        {"g_channel", "1.0"},
        {"b_channel", "1.0"},

        {"fade", "0.006"},
        {"slider", "1.1"},
        {"equation", "0"},
        {"offset1", "0.17"},
        {"offset2", "0.29"},
        {"rotater", "0"},
        
// Raymarching Stuff
        {"pov_xz", "0"},
        {"pov_y", "0"},
        {"pov_q1", "1"},
        {"pov_qi", "0"},
        {"pov_qj", "0"},
        {"pov_qk", "0"},
        {"pov_fov", "3"},
        {"pov_max_dist", "10"},


        {"jj_1", "-1"},
        {"jj_i", "0"},
        {"jj_j", "0"},
        {"jj_ij", "0"},
    });
}


void FourDAlgebraScene::draw() {

    const quat camera_direction_0 = normalize(quat(cos(state["pov_xz"]), 0, sin(state["pov_xz"]), 0));
    const quat camera_direction = camera_direction_0*normalize(quat(cos(state["pov_y"]), sin(state["pov_y"])*sin(state["pov_xz"]), 0, sin(state["pov_y"])*cos(state["pov_xz"])));

    const vec3 camera_pos = rotate_vector(vec3(0,0,-state["pov_max_dist"]*0.5), camera_direction);
    
    
    // M = matrixMult( M, rotationMatrix(4,4,0,1,state["offset"]), 4, 4, 4);
    float **M = rotationMatrix(4,4,0,1,state["offset1"]);
    M = matrixMult( M, rotationMatrix(4,4,2,3,state["offset1"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,0,3,state["offset2"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,1,2,state["offset2"]), 4, 4, 4);

    // float **M = rotationMatrix(4,4,0,3,state["rotation_1k"]);
    M = matrixMult( M, rotationMatrix(4,4,2,3,state["rotation_jk"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,1,3,state["rotation_ik"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,0,3,state["rotation_1k"]), 4, 4, 4);



    four_d_render(get_width_height(),

        camera_direction, 
        camera_pos,
        state["pov_fov"], 
        state["pov_max_dist"],
            
        vec4(M[0][0],M[0][3],M[0][2],M[0][3])*state["scale"],
        vec4(M[1][0],M[1][1],M[1][2],M[1][3])*state["scale"],
        vec4(M[2][0],M[2][3],M[2][2],M[2][3])*state["scale"],

        // vec4(0,1,0,0),
        // vec4(0,0,1,0),
        // vec4(0,0,0,1),

        vec4(cos(state["rotater"]), 0, sin(state["rotater"])*sin(0.4), sin(state["rotater"])*cos(0.4)),
        vec4(cos(state["rotater"]), 0,-sin(state["rotater"])*sin(0.4), -sin(state["rotater"])*cos(0.4)),

        vec4(state["jj_1"], state["jj_i"], state["jj_j"], state["jj_ij"]),
        vec4(-state["jj_i"], state["jj_1"], -state["jj_ij"], state["jj_j"]),

        state["brightness"], 
        vec3(state["r_channel"], state["g_channel"], state["b_channel"]),

        state["fade"], 
        state["slider"], 
        state["equation"],
        gpu_pix.get_ptr()
    );

}
