
#include "TwoDAlgebraScene.h"
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/Color.h"
#include "../../IO/Latex.h"
#include <complex>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

using std::complex;

HOST_DEVICE inline uint32_t OKLABtoRGB(int alpha, float L, float a, float b);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);
extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);
extern "C" void draw_quadrilateral(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const vec2& p3, const uint32_t color);
extern "C" void draw_triangle(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const uint32_t color);
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
        {"dragger_brightness", "0"},
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
        {"xdrag_opacity", "0"},
        {"ydrag_opacity", "0"},
        {"xdrag_color", "1"},
        {"ydrag_color", "1"},
        {"label_type", "0"},
        
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

const string get_label(double label_type, int label_index){
    if (label_type < 1){
        string labels[5] = {"x²","xy","y²","x²","y²"};
        return labels[label_index];

    } else if (label_type < 2){
        string labels[5] = {"","","","1","i"};
        return labels[label_index];

    } else if (label_type < 3){
        string labels[5] = {"1²","","","1i","i²"};
        return labels[label_index];
        
    } else if (label_type < 4){
        string labels[5] = {"","","","x","y"};
        return labels[label_index];
        
    } else if (label_type < 5){
        string labels[5] = {"","","","x²","xy"};
        return labels[label_index];
        
    } else if (label_type < 6){
        string labels[5] = {"","","","xy","y²"};
        return labels[label_index];
        
    }

    return "";

}



void TwoDAlgebraScene::draw() {

    ivec2 wh = get_width_height();
    vec4 dragger_inverse;
    vec2 drag_times_x = vec2(state["dragger_x"]*state["xx_x"]+state["dragger_y"]*state["xy_x"],state["dragger_x"]*state["xx_y"]+state["dragger_y"]*state["xy_y"]);
    vec2 drag_times_y = vec2(state["dragger_x"]*state["xy_x"]+state["dragger_y"]*state["yy_x"],state["dragger_x"]*state["xy_y"]+state["dragger_y"]*state["yy_y"]);

    if (state["number_line"] > 0){
        dragger_inverse = vec4(1/state["dragger_x"],0,0,1);

    } else {
       
        vec4 dragger_calc = vec4(0,0,0,0);
     
        float determinant = drag_times_x.x*drag_times_y.y-drag_times_x.y*drag_times_y.x;
        while (determinant == 0){
            drag_times_x.y += 0.000001;
            drag_times_y.x += 0.000001;
            drag_times_y.y += 0.000001;
            determinant = drag_times_x.x*drag_times_y.y-drag_times_x.y*drag_times_y.x;
        }

        dragger_inverse = vec4(drag_times_y.y, -drag_times_y.x, -drag_times_x.y, drag_times_x.x)/determinant;
    }


    // vec2 origin = wh*0.5;
    // float scalar = wh.y/(state["bottom_y"]-state["top_y"]);
    // for (float x = -10; x < 11; x++){
    //     for (float y = -10; y < 11; y++){
    //         vec2 arrow_end;
    //         if (state["dragger_type"] < 2){
    //             arrow_end = vec2(x+state["dragger_x"], -y-state["dragger_y"]);
    //         } else {
    //             arrow_end = vec2(
    //                 state["dragger_x"]*x*state["xx_x"]+(state["dragger_y"]*x+state["dragger_x"]*y)*state["xy_x"]+state["dragger_y"]*y*state["yy_x"],
    //                 -(state["dragger_x"]*x*state["xx_y"]+(state["dragger_y"]*x+state["dragger_x"]*y)*state["xy_y"]+state["dragger_y"]*y*state["yy_y"])
    //             );
    //         }

    //         uint32_t arrow_color = OKLABtoRGB(255,1,x*0.08,y*0.08);
    //         vec2 arrow_vector = arrow_end-vec2(x,-y);
    //         vec2 arrow_start = vec2(x,-y)*scalar+origin;
    //         arrow_end = arrow_end*scalar+origin;
    //         if (length(arrow_vector)<0.2){
    //             draw_circle(gpu_pix.get_ptr(), get_width_height(), arrow_start, scalar*0.05, arrow_color);
    //         } else {
    //             float angle = atan2(arrow_vector.y,arrow_vector.x);
    //             vec2 arrow_para = vec2(cos(angle),sin(angle))*0.02*scalar;
    //             vec2 arrow_perp = vec2(-sin(angle),cos(angle))*0.02*scalar;
    //             draw_quadrilateral(gpu_pix.get_ptr(), get_width_height(), arrow_start+arrow_perp, arrow_start-arrow_perp, arrow_end-arrow_perp, arrow_end+arrow_perp,  arrow_color);
    //             draw_triangle(gpu_pix.get_ptr(), get_width_height(), arrow_end+arrow_para*4, arrow_end-arrow_para*4+arrow_perp*6, arrow_end-arrow_para*4-arrow_perp*6, arrow_color);

    //         }

    //     }
    // }

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
    

    int diagram_unit = get_diagram_unit(wh,state["top_y"],state["bottom_y"],state["diagram_opacity"]);
    float point_radius = diagram_unit*0.18;
    ivec2 diagram_origin = get_diagram_origin(wh, state["diagram_opacity"], diagram_unit);
    const vec2 textbox_size(point_radius * 3);
    const vec2 textbox_offset = vec2(0,point_radius*0.2);

    const int opacity = ((int) state["diagram_opacity"]) << 24;
    if (opacity != 0){

        int axis_width = wh.y*0.004;
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(0,0), diagram_origin*2, opacity + 0x00000033);
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), diagram_origin-ivec2(axis_width,diagram_unit), diagram_origin+ivec2(axis_width,diagram_unit), opacity + 0x005588cc);
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), diagram_origin-ivec2(diagram_unit,axis_width), diagram_origin+ivec2(diagram_unit,axis_width), opacity + 0x005588cc);
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(diagram_origin.x*2-axis_width,0), diagram_origin*2+axis_width, opacity + 0x005588cc);
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(0,diagram_origin.y*2-axis_width), diagram_origin*2+axis_width, opacity + 0x005588cc);

    }


    const int xx_opacity = ((int) state["xx_opacity"]) << 24;
    if (xx_opacity != 0){
        const vec2 xx_pos = vec2(state["xx_x"], -state["xx_y"])*diagram_unit+diagram_origin;
        draw_circle(gpu_pix.get_ptr(), get_width_height(), xx_pos, point_radius, xx_opacity + 0x00dd44dd);
        write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(xx_opacity, get_label(state["label_type"],0)), xx_pos+textbox_offset*0.5, textbox_size, 1, 0);
    }

    const int yy_opacity = ((int) state["yy_opacity"]) << 24;
    if (yy_opacity != 0){
        const vec2 yy_pos = vec2(state["yy_x"], -state["yy_y"])*diagram_unit+diagram_origin;
        draw_circle(gpu_pix.get_ptr(), get_width_height(), yy_pos, point_radius, yy_opacity + 0x00dddd44);
        write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(yy_opacity, get_label(state["label_type"],2)), yy_pos+textbox_offset, textbox_size, 1, 0);
    }

    const int xy_opacity = ((int) state["xy_opacity"]) << 24;
    if (xy_opacity != 0){
        const vec2 xy_pos = vec2(state["xy_x"], -state["xy_y"])*diagram_unit+diagram_origin;
        draw_circle(gpu_pix.get_ptr(), get_width_height(), xy_pos, point_radius, xy_opacity + 0x00ccccee);
        write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(xy_opacity, get_label(state["label_type"],1)), xy_pos+textbox_offset, textbox_size, 1, 0);
    }
    
    const int xdrag_opacity = ((int) state["xdrag_opacity"]) << 24;
    if (xdrag_opacity != 0){
        const vec2 xdrag_pos = drag_times_x*vec2(1,-1)*diagram_unit+diagram_origin;
        draw_circle(gpu_pix.get_ptr(), get_width_height(), xdrag_pos, point_radius, xdrag_opacity+colorlerp(0x00ccccee,0x00dd44dd,state["xdrag_color"]));
        write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(xdrag_opacity, get_label(state["label_type"],3)), xdrag_pos+textbox_offset*0.5, textbox_size, 1, 0);
    }

    const int ydrag_opacity = ((int) state["ydrag_opacity"]) << 24;
    if (ydrag_opacity != 0){
        const vec2 ydrag_pos = drag_times_y*vec2(1,-1)*diagram_unit+diagram_origin;
        draw_circle(gpu_pix.get_ptr(), get_width_height(), ydrag_pos, point_radius, ydrag_opacity+colorlerp(0x00ccccee,0x00dddd44,state["ydrag_color"]));
        write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(ydrag_opacity, get_label(state["label_type"],4)), ydrag_pos+textbox_offset*0.5, textbox_size, 1, 0);
    }

}

