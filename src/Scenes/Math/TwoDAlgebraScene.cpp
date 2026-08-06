
#include "TwoDAlgebraScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

#include <vector>
#include <stdexcept>
#include <string>

using std::complex;

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);
extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);
extern "C" void two_d_algebra(
    uint32_t* d_pixels, const ivec2& wh,
    // vec2 dragger, 
    vec2 dragger_pos, 
    float dragger_type, float dragger_brightness, 
    vec4 dragger_inverse,

    float number_line, int brightness,
    const vec2& lx_ty, const vec2& rx_by
);



TwoDAlgebraScene::TwoDAlgebraScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        {"dragger_x", "0"},
        {"dragger_y", "0"},
        {"dragger_type", "0"},
        {"algebra", "2"},
        {"number_line", "0"},
        {"brightness", "255"},
        {"dragger_brightness", "1"},
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"yx_x", "0"},
        {"yx_y", "1"},
        {"yy_x", "-1"},
        {"yy_y", "0"},

        {"diagram_opacity", "0"},
        {"diagram_xx", "0"},
        {"diagram_xy", "0"},
        {"diagram_yy", "0"},
        
    });
}

const StateQuery TwoDAlgebraScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {
        "dragger_x", "dragger_y", "dragger_type","dragger_brightness","algebra",
        "number_line","brightness",
        "xx_x", "xx_y","xy_x", "xy_y","yx_x", "yx_y","yy_x", "yy_y",
        "diagram_opacity","diagram_xx","diagram_xy","diagram_yy"
    });
    return sq;
}


void TwoDAlgebraScene::draw() {


    vec4 dragger_inverse;
    if (state["number_line"] > 0){
        dragger_inverse = vec4(1/state["dragger_x"],0,0,1);

    } else {
       
        vec4 dragger_calc = vec4(0,0,0,0);
        vec2 drag_times_x = vec2(state["dragger_x"]*state["xx_x"]+state["dragger_y"]*state["xy_x"],state["dragger_x"]*state["xx_y"]+state["dragger_y"]*state["xy_y"]);
        vec2 drag_times_y = vec2(state["dragger_x"]*state["xy_x"]+state["dragger_y"]*state["yy_x"],state["dragger_x"]*state["xy_y"]+state["dragger_y"]*state["yy_y"]);

        float determinant = drag_times_x.x*drag_times_y.y-drag_times_x.y*drag_times_y.x;
        while (determinant == 0){
            drag_times_x.y += 0.000001;
            drag_times_y.x += 0.000001;
            drag_times_y.y += 0.000001;
            determinant = drag_times_x.x*drag_times_y.y-drag_times_x.y*drag_times_y.x;
        }

        dragger_inverse = vec4(drag_times_y.y, -drag_times_y.x, -drag_times_x.y, drag_times_x.x)/determinant;
    }

    two_d_algebra(
        gpu_pix->get_ptr(), get_width_height(),
        //dragger_calc, 
        vec2(state["dragger_x"], state["dragger_y"]), 
        state["dragger_type"], state["dragger_brightness"], 
        // state["algebra"],
        dragger_inverse,


        state["number_line"],
        int(state["brightness"]) << 24,

        vec2(state["left_x"], state["top_y"]),
        vec2(state["right_x"], state["bottom_y"])

        
    );


    vec2 wh = get_width_height();
    int diagram_unit = wh.y*0.1;
    int axis_width = wh.y*0.004;
    float point_radius = diagram_unit*0.18;
    ivec2 diagram_origin = ivec2(diagram_unit*1.6,diagram_unit*1.6);

    int opacity = ((int) state["diagram_opacity"]) << 24;

    draw_rectangle(gpu_pix->get_ptr(), get_width_height(), ivec2(0,0), diagram_origin*2, opacity + 0x00000033);
    draw_rectangle(gpu_pix->get_ptr(), get_width_height(), diagram_origin-ivec2(axis_width,diagram_unit), diagram_origin+ivec2(axis_width,diagram_unit), opacity + 0x005588cc);
    draw_rectangle(gpu_pix->get_ptr(), get_width_height(), diagram_origin-ivec2(diagram_unit,axis_width), diagram_origin+ivec2(diagram_unit,axis_width), opacity + 0x005588cc);
    draw_rectangle(gpu_pix->get_ptr(), get_width_height(), ivec2(diagram_origin.x*2-axis_width,0), diagram_origin*2+axis_width, opacity + 0x005588cc);
    draw_rectangle(gpu_pix->get_ptr(), get_width_height(), ivec2(0,diagram_origin.y*2-axis_width), diagram_origin*2+axis_width, opacity + 0x005588cc);
    
    draw_circle(gpu_pix->get_ptr(), get_width_height(), vec2(state["xx_x"], -state["xx_y"])*diagram_unit+diagram_origin, point_radius, opacity + 0x00dd44dd);
    draw_circle(gpu_pix->get_ptr(), get_width_height(), vec2(state["yy_x"], -state["yy_y"])*diagram_unit+diagram_origin, point_radius, opacity + 0x00dddd44);
    draw_circle(gpu_pix->get_ptr(), get_width_height(), vec2(state["xy_x"], -state["xy_y"])*diagram_unit+diagram_origin, point_radius, opacity + 0x00ccccee);


}


