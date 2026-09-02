
#include "FourDPlaneScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

using std::complex;

extern "C" void four_d_plane_render(
    const ivec2& wh,
    const vec2& lx_ty,
    const vec2& rx_by,
    vec4 x_unit,
    vec4 y_unit,

    vec4 jj,
    vec4 ijj,
    const float brightness,
    const vec3 channels,
    unsigned int internal_color,
    unsigned int* d_colors
);

FourDPlaneScene::FourDPlaneScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        {"rotation_1k", "0"},
        {"rotation_ik", "0"},
        {"rotation_jk", "0"},
        {"scale", "1.0"},

        {"offset1", "0.0"},
        {"offset2", "0.0"},

        {"brightness", "1.0"},
        {"r_channel", "1.0"},
        {"g_channel", "1.0"},
        {"b_channel", "1.0"},
        
        {"jj_1", "-1"},
        {"jj_i", "0"},
        {"jj_j", "0"},
        {"jj_ij", "0"},
    });
}

// float **newMatrix(int rows, int cols){
//     float **M = new float *[rows];
//     for (int r = 0; r < rows; r++){
//         M[r] = new float[cols]{0.0};
//     }
//     return M;
// }

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

void FourDPlaneScene::draw() {

    
    // M = matrixMult( M, rotationMatrix(4,4,0,1,state["offset"]), 4, 4, 4);
    float **M = rotationMatrix(4,4,0,1,state["offset1"]);
    M = matrixMult( M, rotationMatrix(4,4,2,3,state["offset1"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,0,3,state["offset2"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,1,2,state["offset2"]), 4, 4, 4);

    // float **M = rotationMatrix(4,4,0,3,state["rotation_1k"]);
    M = matrixMult( M, rotationMatrix(4,4,2,3,state["rotation_jk"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,1,3,state["rotation_ik"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,0,3,state["rotation_1k"]), 4, 4, 4);

    four_d_plane_render(get_width_height(),

        vec2(state["left_x"], state["top_y"]),
        vec2(state["right_x"], state["bottom_y"]), 
            
        vec4(M[2][0],M[2][3],M[2][2],M[2][3])*state["scale"],
        vec4(M[1][0],M[1][1],M[1][2],M[1][3])*state["scale"],

        vec4(state["jj_1"], state["jj_i"], state["jj_j"], state["jj_ij"]),
        vec4(-state["jj_i"], state["jj_1"], -state["jj_ij"], state["jj_j"]),

        state["brightness"], 
        vec3(state["r_channel"], state["g_channel"], state["b_channel"]),

        OPAQUE_BLACK,
        gpu_pix.get_ptr()
    );

    CoordinateScene::draw();
}
