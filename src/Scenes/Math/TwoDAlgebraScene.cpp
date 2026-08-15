
#include "TwoDAlgebraScene.h"
#include "../../Host_Device_Shared/vec.h"
#include "../../IO/Latex.h"
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
        {"xx_opacity", "0"},
        {"xy_opacity", "0"},
        {"yy_opacity", "0"},
        
    });
}

const int get_diagram_unit(ivec2 wh, float top_y, float bottom_y, int diagram_opacity){
    
    if (diagram_opacity == 255){
        return wh.y*0.1;
    } else if (diagram_opacity == 0) {
        return wh.y/(bottom_y-top_y);
    } else {
        return int(get_diagram_unit(wh, top_y, bottom_y, 0)*(1.0-diagram_opacity/255.0) + get_diagram_unit(wh, top_y, bottom_y, 255)*diagram_opacity/255.0);
    }
}

const ivec2 get_diagram_origin(ivec2 wh, int diagram_opacity, int diagram_unit){
    
    if (diagram_opacity == 255){
        return ivec2(diagram_unit*1.6,diagram_unit*1.6);
    } else if (diagram_opacity == 0) {
        return ivec2(wh*0.5);
    } else {
        return ivec2(get_diagram_origin(wh, 0, diagram_unit)*(1-diagram_opacity/255.0) + get_diagram_origin(wh, 255, diagram_unit)*diagram_opacity/255.0);
    }
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
        gpu_pix.get_ptr(), get_width_height(),
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
    

    ivec2 wh = get_width_height();
    int diagram_unit = get_diagram_unit(wh,state["top_y"],state["bottom_y"],state["diagram_opacity"]);
    int axis_width = wh.y*0.004;
    float point_radius = diagram_unit*0.2;
    ivec2 diagram_origin = get_diagram_origin(wh, state["diagram_opacity"], diagram_unit);

    int opacity = ((int) state["diagram_opacity"]) << 24;

    draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(0,0), diagram_origin*2, opacity + 0x00000033);
    draw_rectangle(gpu_pix.get_ptr(), get_width_height(), diagram_origin-ivec2(axis_width,diagram_unit), diagram_origin+ivec2(axis_width,diagram_unit), opacity + 0x005588cc);
    draw_rectangle(gpu_pix.get_ptr(), get_width_height(), diagram_origin-ivec2(diagram_unit,axis_width), diagram_origin+ivec2(diagram_unit,axis_width), opacity + 0x005588cc);
    draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(diagram_origin.x*2-axis_width,0), diagram_origin*2+axis_width, opacity + 0x005588cc);
    draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(0,diagram_origin.y*2-axis_width), diagram_origin*2+axis_width, opacity + 0x005588cc);

    const vec2 xx_pos = vec2(state["xx_x"], -state["xx_y"])*diagram_unit+diagram_origin;
    const vec2 yy_pos = vec2(state["yy_x"], -state["yy_y"])*diagram_unit+diagram_origin;
    const vec2 xy_pos = vec2(state["xy_x"], -state["xy_y"])*diagram_unit+diagram_origin;
    const int xx_opacity = ((int) state["xx_opacity"]) << 24;
    const int xy_opacity = ((int) state["xy_opacity"]) << 24;
    const int yy_opacity = ((int) state["yy_opacity"]) << 24;
    draw_circle(gpu_pix.get_ptr(), get_width_height(), xx_pos, point_radius, xx_opacity + 0x00dd44dd);
    draw_circle(gpu_pix.get_ptr(), get_width_height(), yy_pos, point_radius, xy_opacity + 0x00dddd44);
    draw_circle(gpu_pix.get_ptr(), get_width_height(), xy_pos, point_radius, yy_opacity + 0x00ccccee);

    const vec2 textbox_size(point_radius * 3);
    const vec2 textbox_offset = vec2(0,point_radius*0.2);
    write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(xx_opacity, "xx"), xx_pos+textbox_offset*0.5, textbox_size, 1, 0);
    write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(xy_opacity, "yy"), yy_pos+textbox_offset, textbox_size, 1, 0);
    write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(yy_opacity, "xy"), xy_pos+textbox_offset, textbox_size, 1, 0);
}

