
#include "FourDPlaneScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

using std::complex;

extern "C" void four_d_render(
    const ivec2& wh,
    const vec2& lx_ty,
    const vec2& rx_by,
    vec4 x_unit,
    vec4 y_unit,
    const float brightness,
    unsigned int internal_color,
    unsigned int* d_colors
);

FourDPlaneScene::FourDPlaneScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        {"rotation_1", "0"},
        {"rotation_2", "0"},
        {"rotation_3", "0"},
        {"scale", "1.0"},
        {"brightness", "1.0"}
        
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

const StateQuery FourDPlaneScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {"rotation_1", "rotation_2", "rotation_3", "scale", "brightness"});
    return sq;
}

void FourDPlaneScene::draw() {

    float **M = rotationMatrix(4,4,0,1,state["rotation_1"]);
    M = matrixMult( M, rotationMatrix(4,4,2,3,state["rotation_2"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,1,2,state["rotation_3"]), 4, 4, 4);
    M = matrixMult( M, rotationMatrix(4,4,0,3,state["rotation_1"]), 4, 4, 4);
    // M = matrixMult( M, rotationMatrix(4,4,0,2,state["rotation_2"]), 4, 4, 4);
    // M = matrixMult( M, rotationMatrix(4,4,1,3,state["rotation_3"]), 4, 4, 4);
    // float **M3 = rotationMatrix(4,4,0,2,state["rotation_3"]);
    // float **M3 = rotationMatrix(4,4,1,2,state["rotation_1"]);
    // float **M2 = rotationMatrix(4,4,1,3,state["rotation_2"]);
    // float **M3 = rotationMatrix(4,4,0,3,state["rotation_3"]);
    // float **M = matrixMult(matrixMult(M1,M2,4,4,4),M3,4,4,4);

    four_d_render(get_width_height(),
                      vec2(state["left_x"], state["top_y"]),
                      vec2(state["right_x"], state["bottom_y"]),
                      
                      vec4(M[0][0],M[0][3],M[0][2],M[0][3])*state["scale"],
                      vec4(M[1][0],M[1][1],M[1][2],M[1][3])*state["scale"],
                    //   vec4(M[2][0],M[2][3],M[2][2],M[2][3])*state["scale"],
                    //   vec4(M[3][0],M[3][1],M[3][2],M[3][3])*state["scale"],
                    //   vec4(8,8,8,8),
                    //   vec4(8,8,8,8),
                      state["brightness"], 
                      OPAQUE_BLACK,
                      gpu_pix->get_ptr()
    );

    CoordinateScene::draw();
}
