
#include "TwoDAlgebraScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

#include <vector>
#include <stdexcept>
#include <string>

using std::complex;

// extern "C" void two_d_algebra(
//     const ivec2& wh,
//     const vec2& lx_ty,
//     const vec2& rx_by,
//     ResolvedStateEquationComponent* x_eq, 
//     int x_eq_size,
//     ResolvedStateEquationComponent* y_eq,
//     int y_eq_size,
//     // vec4 x_unit,
//     // vec4 y_unit,
//     // const float brightness,
//     // unsigned int internal_color,
//     unsigned int* d_colors
// );

extern "C" void two_d_algebra(
    uint32_t* d_pixels, const ivec2& wh,
    ResolvedStateEquationComponent* x_eq, int x_eq_size, float x_adjustment,
    ResolvedStateEquationComponent* y_eq, int y_eq_size, float y_adjustment,
    float dragger_x, float dragger_y,
    const vec2& lx_ty, const vec2& rx_by
);


TwoDAlgebraScene::TwoDAlgebraScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        {"x_equation", "1"},
        {"y_equation", "0"},
        {"x_adjustment", "1"},
        {"y_adjustment", "1"},
        {"dragger_x", "0"},
        {"dragger_y", "0"}
        
    });
}

const StateQuery TwoDAlgebraScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {
        "x_equation", "y_equation", 
        "x_adjustment", "y_adjustment",
        "dragger_x", "dragger_y"
    });
    return sq;
}


void TwoDAlgebraScene::draw() {
    vector<ResolvedStateEquationComponent> x_eq = manager.get_resolved_equation("x_equation");
    vector<ResolvedStateEquationComponent> y_eq = manager.get_resolved_equation("y_equation");
    // vector<ResolvedStateEquationComponent> r = manager.get_resolved_equation("x_equation");

    // float **M = rotationMatrix(4,4,0,1,state["rotation_1"]);
    // float **M2 = rotationMatrix(4,4,1,3,state["rotation_2"]);
    // float **M3 = rotationMatrix(4,4,0,3,state["rotation_3"]);
    // float **M = matrixMult(matrixMult(M1,M2,4,4,4),M3,4,4,4);

    two_d_algebra(
        gpu_pix->get_ptr(), get_width_height(),
        x_eq.data(), x_eq.size(), state["x_adjustment"],
        y_eq.data(), y_eq.size(), state["y_adjustment"],
        state["dragger_x"], state["dragger_y"],
        vec2(state["left_x"], state["top_y"]),
        vec2(state["right_x"], state["bottom_y"])
        
    //   vec4(M[0][0],M[0][3],M[0][2],M[0][3])*state["scale"],
    //   vec4(8,8,8,8),
    //   state["brightness"], 
    //   OPAQUE_BLACK,
    );


    // two_d_algebra(
    //     gpu_pix->get_ptr(), get_width_height(),
    //     r.data(), r.size(),
    //     vec2(state["left_x"], state["top_y"]),
    //     vec2(state["right_x"], state["bottom_y"])
    // );
}


