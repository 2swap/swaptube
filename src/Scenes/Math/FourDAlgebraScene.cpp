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

    const float brightness,
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
        {"pov_max_dist", "10"}
    });
}

// float **newMatrix(int rows, int cols){
//     float **M = new float *[rows];
//     for (int r = 0; r < rows; r++){
//         M[r] = new float[cols]{0.0};
//     }
//     return M;
// }
/*
float **rotationMatrix(int rows, int cols, int axis1, int axis2, float angle){
    float **M = new float *[rows];

    for (int r = 0; r < rows; r++){
        M[r] = new float[cols]{0.0};

        for (int c = 0; c < cols; c++){
            if ((r == axis1 && c == axis2)){
                M[r][c] = sin(angle);
            } else if (r == axis1 && c == axis2){
                M[r][c] = -sin(angle);
            } else if (r != c){
                M[r][c] = 0.0;
            } else if (c == axis1 || r == axis2){
                M[r][c] = cos(angle);
            } else {
                M[r][c] = 1.0;
            }
        }
    }
    return M;
}


float **matrixMult(float **A,float **B, int rows, int cols, int shared){

    float **AB = new float *[rows];

    for (int r = 0; r < rows; r++){
        AB[r] = new float[cols]{0.0};

        for (int c = 0; c < cols; c++){
            for (int s = 0; s < shared; s++){
                AB[r][c] += A[r][s]*B[s][c];
            }
        }
    }

	return AB;
}
*/
const StateQuery FourDAlgebraScene::populate_state_query() const {
    return {
    
        "rotation_1k", "rotation_ik", "rotation_jk", 
        "scale", "brightness","fade","slider",
        "equation","offset1","offset2",
        "rotater",
        "pov_xz", "pov_y",// "pov_z",
        "pov_q1", "pov_qi", "pov_qj", "pov_qk",
        "pov_fov", "pov_max_dist"
    };
}

void FourDAlgebraScene::draw() {

    // vec3 camera_pos = vec3(sin(state["pov_xz"])*4, cos(state["pov_xz"])*4, state["pov_y"]);
    // vec3 camera_pos = vec3(0,-5,0);
    // quat camera_direction = normalize(quat(state["pov_q1"], state["pov_qi"], state["pov_qj"], state["pov_qk"]));


    const quat camera_direction_0 = normalize(quat(cos(state["pov_xz"]), 0, sin(state["pov_xz"]), 0));
    const quat camera_direction = camera_direction_0*normalize(quat(cos(state["pov_y"]), sin(state["pov_y"])*sin(state["pov_xz"]), 0, sin(state["pov_y"])*cos(state["pov_xz"])));

    // const quat camera_direction = normalize(quat(cos(state["pov_xz"]), sin(state["pov_xz"]), 0, 0));

    
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



    // float **M = rotationMatrix(4,4,2,3,state["rotation_1"]);
    // M = matrixMult( M, rotationMatrix(4,4,2,3,state["rotation_2"]), 4, 4, 4);
    // M = matrixMult( M, rotationMatrix(4,4,1,2,state["rotation_3"]), 4, 4, 4);
    // M = matrixMult( M, rotationMatrix(4,4,0,3,state["rotation_1"]), 4, 4, 4);
    // M = matrixMult( M, rotationMatrix(4,4,0,2,state["rotation_2"]), 4, 4, 4);
    // M = matrixMult( M, rotationMatrix(4,4,1,3,state["rotation_3"]), 4, 4, 4);
    // float **M3 = rotationMatrix(4,4,0,2,state["rotation_3"]);
    // float **M3 = rotationMatrix(4,4,1,2,state["rotation_1"]);
    // float **M2 = rotationMatrix(4,4,1,3,state["rotation_2"]);
    // float **M3 = rotationMatrix(4,4,0,3,state["rotation_3"]);
    // float **M = matrixMult(matrixMult(M1,M2,4,4,4),M3,4,4,4);

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

        // vec4(M[3][0],M[3][1],M[3][2],M[3][3])*state["scale"],
        //   vec4(8,8,8,8),
        //   vec4(8,8,8,8),
        state["brightness"], 
        state["fade"], 
        state["slider"], 
        state["equation"],
        gpu_pix->get_ptr()
    );

}
