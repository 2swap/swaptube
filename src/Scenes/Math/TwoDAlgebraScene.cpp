
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
    // ResolvedStateEquationComponent* x_eq, int x_eq_size, float x_adjustment,
    // ResolvedStateEquationComponent* y_eq, int y_eq_size, float y_adjustment,
    vec2 dragger, vec2 dragger_pos, float dragger_type, float algebra,
    // vec2 xx, vec2 xy, vec2 yx, vec2 yy,
    float number_line, int brightness,
    const vec2& lx_ty, const vec2& rx_by
);



TwoDAlgebraScene::TwoDAlgebraScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        // {"x_equation", "1"},
        // {"y_equation", "0"},
        // {"x_adjustment", "1"},
        // {"y_adjustment", "1"},
        {"dragger_x", "0"},
        {"dragger_y", "0"},
        {"dragger_type", "0"},
        {"algebra", "2"},
        {"number_line", "0"},
        {"brightness", "255"},
        // {"xx_x", "1"},
        // {"xx_y", "0"},
        // {"xy_x", "0"},
        // {"xy_y", "1"},
        // {"yx_x", "0"},
        // {"yx_y", "1"},
        // {"yy_x", "-1"},
        // {"yy_y", "0"},
        
    });
}

const StateQuery TwoDAlgebraScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {
        "dragger_x", "dragger_y", "dragger_type","algebra",
        "number_line","brightness",
        // "xx_x", "xx_y","xy_x", "xy_y","yx_x", "yx_y","yy_x", "yy_y",
    });
    return sq;
}

/*
algebra codes:
     0 - R^2
     1 - C
     2 - xx = 0, xy = -x-y, yy = y
     3 - duals
     4 - splitties

*/

void TwoDAlgebraScene::draw() {
    // vector<ResolvedStateEquationComponent> x_eq = manager.get_resolved_equation("x_equation");
    // vector<ResolvedStateEquationComponent> y_eq = manager.get_resolved_equation("y_equation");

    vec2 dragger_calc = vec2(0,0);
    if (state["dragger_type"] == 1){
        dragger_calc = vec2(-state["dragger_x"], -state["dragger_y"]);
        
    } else if (state["dragger_type"] == 2){

        if (state["number_line"] > 0){
            dragger_calc = vec2(1/state["dragger_x"], 1);
            
        } else if (state["algebra"] == 0){
            dragger_calc = vec2(1/state["dragger_x"], 1/max(state["dragger_y"],0.001));

        } else if (state["algebra"] == 1){
            float drag_size = state["dragger_x"]*state["dragger_x"] + state["dragger_y"]*state["dragger_y"];
            dragger_calc = vec2(state["dragger_x"]/drag_size, -state["dragger_y"]/drag_size);

        } else if (state["algebra"] == 2){
            dragger_calc = vec2(state["dragger_x"], state["dragger_y"]);

        } else if (state["algebra"] == 3){
            float drag_size = max(state["dragger_x"]*state["dragger_x"],0.001);
            dragger_calc = vec2(state["dragger_x"]/drag_size, -state["dragger_y"]/drag_size);

        } else if (state["algebra"] == 4){
            float drag_size = state["dragger_x"]*state["dragger_x"] - state["dragger_y"]*state["dragger_y"];
            if (abs(drag_size) < 0.001){
                drag_size = 0.001;
            }
            dragger_calc = vec2(state["dragger_x"]/drag_size, -state["dragger_y"]/drag_size);

        } 
    }


    two_d_algebra(
        gpu_pix->get_ptr(), get_width_height(),
        // x_eq.data(), x_eq.size(), state["x_adjustment"],
        // y_eq.data(), y_eq.size(), state["y_adjustment"],
        dragger_calc, 
        vec2(state["dragger_x"], state["dragger_y"]), 
        state["dragger_type"], state["algebra"],

        // vec2(state["xx_x"], state["xx_y"]),
        // vec2(state["xy_x"], state["xy_y"]),
        // vec2(state["yx_x"], state["yx_y"]),
        // vec2(state["yy_x"], state["yy_y"]),

        state["number_line"],
        int(state["brightness"]) << 24,
        vec2(state["left_x"], state["top_y"]),
        vec2(state["right_x"], state["bottom_y"])

        
    );


    // two_d_algebra(
    //     gpu_pix->get_ptr(), get_width_height(),
    //     r.data(), r.size(),
    //     vec2(state["left_x"], state["top_y"]),
    //     vec2(state["right_x"], state["bottom_y"])
    // );
}


