
#include "TwoDAlgebraScene.h"
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/Color.h"
#include "../../Host_Device_Shared/helpers.h"
#include "../../Core/Pixels.h"
#include "../../IO/Latex.h"
#include <complex>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

using std::complex;

HOST_DEVICE inline uint32_t OKLABtoRGB(int alpha, float L, float a, float b);
extern "C" void cuda_render_many_lines_from_host(uint32_t* d_pixels, const ivec2& wh, const vec2* h_line_list, const int line_count,
    const vec2& lx_ty, const vec2& rx_by, const uint32_t* colors, const float opacity, const float thickness);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);
extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);
extern "C" void draw_quadrilateral(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const vec2& p3, const uint32_t color);
extern "C" void draw_triangle(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const uint32_t color);
extern "C" void two_d_algebra(
    uint32_t* d_pixels, const ivec2& wh,
    int equation,
    float equation_lerp,
    vec2 channels,
    vec2 xx,  vec2 xy,  vec2 yy, 
    int brightness,
    const vec2& lx_ty, const vec2& rx_by
);



TwoDAlgebraScene::TwoDAlgebraScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        {"dragger_x", "0"},{"dragger_y", "0"},
        {"operator_x", "0"},{"operator_y", "0"},
        {"dragger_type", "0"},
        {"dragger_shown", "0"},
        {"dragger_active", "1"},
        {"algebra", "2"},
        {"number_line", "0"},
        {"brightness", "255"},
        {"xx_x", "1"},{"xx_y", "0"},{"xy_x", "0"},{"xy_y", "1"},{"yy_x", "-1"},{"yy_y", "0"},
        {"xx_opacity", "0"},{"xy_opacity", "0"},{"yy_opacity", "0"},
        {"grid_scale", "1"},

        {"mode", "0"}, // 0 for grid, other for equation

        {"diagram_opacity", "0"},{"diagram_label", "0"},
        {"x_label", "1"},{"y_label", "1"},
        {"re_channel", "1"},
        {"im_channel", "1"},
        {"point_count", "0"},{"point_label", "0"},{"point_opacity", "255"},{"point_size", "1"},
        {"point_x", "0"},{"point_y", "0"},
        {"force_complex", "0"},

        {"bg_1_r", "0"},{"bg_1_g", "0"},{"bg_1_b", "68"},
        
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

const vec2 two_d_transform(vec2 input_point, vec2 drag_type, vec2 drag_pos, vec2 drag_times_x, vec2 drag_times_y){

    if (drag_type.x == 0){
        return input_point;

    } else if (drag_type.x == 2){ 
        return (input_point.x*drag_times_x + input_point.y*drag_times_y);

    } else if (drag_type.y != 0){ 
        return vec2(input_point.x+drag_pos.x,0);

    } else { 
        return (input_point+drag_pos);

    }
}

const int two_d_color(float x_color, float y_color, int opacity){
    return opacity + OKLABtoRGB(0,1,x_color*0.06,y_color*0.06);
}



void TwoDAlgebraScene::draw() {

    const ivec2 wh = get_width_height();
    const float screen_unit = wh.y/(state["bottom_y"]-state["top_y"]);
    const ivec2 origin = wh*0.5;
    const int bg_color = ((int) state["bg_1_r"]) << 16 | ((int) state["bg_1_g"]) << 8 | (int) state["bg_1_b"];


    vec2 drag_times_x;
    vec2 drag_times_y;
    vec2 drag_pos(get_global_state("dragger_x_6"),get_global_state("dragger_y_6"));

    if (state["dragger_active"] != 0){
        set_global_state("dragger_x_6", get_global_state("dragger_x_5"));
        set_global_state("dragger_y_6", get_global_state("dragger_y_5"));
        set_global_state("dragger_x_5", get_global_state("dragger_x_4"));
        set_global_state("dragger_y_5", get_global_state("dragger_y_4"));
        set_global_state("dragger_x_4", get_global_state("dragger_x_3"));
        set_global_state("dragger_y_4", get_global_state("dragger_y_3"));
        set_global_state("dragger_x_3", get_global_state("dragger_x_2"));
        set_global_state("dragger_y_3", get_global_state("dragger_y_2"));
        set_global_state("dragger_x_2", get_global_state("dragger_x_1"));
        set_global_state("dragger_y_2", get_global_state("dragger_y_1"));
        set_global_state("dragger_x_1", state["dragger_x"]);
        set_global_state("dragger_y_1", state["dragger_y"]);
    } else {
        drag_pos = vec2(state["dragger_x"],state["dragger_y"]);
    }


    if (state["force_complex"] == 0){
        drag_times_x = vec2(drag_pos.x*state["xx_x"]+drag_pos.y*state["xy_x"],drag_pos.x*state["xx_y"]+drag_pos.y*state["xy_y"]);
        drag_times_y = vec2(drag_pos.x*state["xy_x"]+drag_pos.y*state["yy_x"],drag_pos.x*state["xy_y"]+drag_pos.y*state["yy_y"]);
    } else {
        drag_times_x = vec2(drag_pos.x,drag_pos.y);
        drag_times_y = vec2(-drag_pos.y,drag_pos.x);
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
    //         if (length(arrow_vector)<0.2){
    //             draw_circle(gpu_pix.get_ptr(), wh, arrow_start, scalar*0.05, arrow_color, 1.0);
    //         } else {
    //             arrow_end = (vec2(x,-y)+arrow_vector*min(1.0f,1.5f/length(arrow_vector)))*scalar+origin;
    //             float angle = atan2(arrow_vector.y,arrow_vector.x);
    //             vec2 arrow_para = vec2(cos(angle),sin(angle))*0.02*scalar;
    //             vec2 arrow_perp = vec2(-sin(angle),cos(angle))*0.02*scalar;
    //             draw_quadrilateral(gpu_pix.get_ptr(), wh, arrow_start+arrow_perp, arrow_start-arrow_perp, arrow_end-arrow_perp, arrow_end+arrow_perp,  arrow_color);
    //             draw_triangle(gpu_pix.get_ptr(), wh, arrow_end+arrow_para*4, arrow_end-arrow_para*4+arrow_perp*6, arrow_end-arrow_para*4-arrow_perp*6, arrow_color);

    //         }

    //     }
    // }


    if (state["mode"] == 0){

        vector<vec2> point_list;
        vector<uint32_t> color_list;

        const int grid_size_x = 9;
        const int grid_size_y = (state["number_line"]==0) ? grid_size_x : 0;
        const int grid_opacity = ((int) state["brightness"]) << 24;
        const float line_thickness = screen_unit*0.03*state["grid_scale"];
        const float point_size = screen_unit*0.12*state["grid_scale"];
        const float lerp_step = 0.005;

        const vec2 drag_type = vec2(state["dragger_type"],state["number_line"]);


        for (int y = -grid_size_y; y <= grid_size_y; y++){

            vec2 current_pos = two_d_transform(vec2(-grid_size_x,y), 
                drag_type, drag_pos, drag_times_x, drag_times_y);
            vec2 end_pos = two_d_transform(vec2(grid_size_x,y), 
                drag_type, drag_pos, drag_times_x, drag_times_y);

            vec2 vec_diff = (end_pos-current_pos)*lerp_step;

            for (float l = 0; l <= 1; l += lerp_step){

                point_list.push_back(current_pos);
                point_list.push_back(current_pos+vec_diff);
                color_list.push_back(two_d_color(lerp(-grid_size_x,grid_size_x,l),y,grid_opacity));
            
                current_pos = current_pos+vec_diff;
            }
        }

        if (state["number_line"]==0){
            for (int x = -grid_size_x; x <= grid_size_x; x++){
                vec2 current_pos = two_d_transform(vec2(x,-grid_size_y), 
                    drag_type, drag_pos, drag_times_x, drag_times_y);
                vec2 end_pos = two_d_transform(vec2(x,grid_size_y), 
                    drag_type, drag_pos, drag_times_x, drag_times_y);


                vec2 vec_diff = (end_pos-current_pos)*lerp_step;

                for (float l = 0; l <= 1; l += lerp_step){
            
                    point_list.push_back(current_pos);
                    point_list.push_back(current_pos+vec_diff);
                    color_list.push_back(two_d_color(x,lerp(-grid_size_y,grid_size_y,l),grid_opacity));
                
                    current_pos = current_pos+vec_diff;
                }
            }
        }
        
        cuda_render_many_lines_from_host(gpu_pix.get_ptr(), wh, 
            point_list.data(), point_list.size()/2,
            vec2(state["left_x"],state["top_y"]), vec2(state["right_x"],state["bottom_y"]), 
            // 0xffffffff, 1.0,
            color_list.data(), state["brightness"]/255,
            line_thickness
        );


        for (int x = -grid_size_x; x <= grid_size_x; x++){
            for (int y = -grid_size_y; y <= grid_size_y; y++){
                vec2 point_pos = two_d_transform(vec2(x,y), drag_type, drag_pos, drag_times_x, drag_times_y)*vec2(1,-1)*screen_unit+origin;
                draw_circle(gpu_pix.get_ptr(), wh, point_pos, point_size, 
                    two_d_color(x,y,grid_opacity), 1.0);
            }
        }



    } else {

        int equation = (int) state["mode"];
        float equation_lerp = state["mode"]-equation;
        two_d_algebra(
            gpu_pix.get_ptr(), wh,

            equation,
            equation_lerp,
            vec2(state["re_channel"], state["im_channel"]),
            vec2(state["xx_x"], state["xx_y"]),
            vec2(state["xy_x"], state["xy_y"]),
            vec2(state["yy_x"], state["yy_y"]),

            int(state["brightness"]),

            vec2(state["left_x"], state["top_y"]),
            vec2(state["right_x"], state["bottom_y"])
            
        );
    }




    const int dragger_opacity = 0xff000000;//((int) (state["dragger_brightness"]*255)) << 24;
    const ivec2 drag_pixel = ivec2((int) (state["dragger_x"]*screen_unit),(int) (-state["dragger_y"]*screen_unit))+origin;
    const float dragger_size = screen_unit*0.35*sin(state["dragger_shown"]*1.98);

    if (state["dragger_type"] == 1 && state["dragger_shown"] != 0){
        ivec2 pos_diff((int) (dragger_size),(int) (dragger_size*0.4));
        draw_rectangle(gpu_pix.get_ptr(), wh, drag_pixel-pos_diff, drag_pixel+pos_diff, dragger_opacity + bg_color);
        pos_diff = ivec2(pos_diff.y, pos_diff.x);
        draw_rectangle(gpu_pix.get_ptr(), wh, drag_pixel-pos_diff, drag_pixel+pos_diff, dragger_opacity + bg_color);

        pos_diff = ivec2((int) (dragger_size*0.8),(int) (dragger_size*0.2));
        draw_rectangle(gpu_pix.get_ptr(), wh, drag_pixel-pos_diff, drag_pixel+pos_diff, dragger_opacity + 0x00ffffff);
        pos_diff = ivec2(pos_diff.y, pos_diff.x);
        draw_rectangle(gpu_pix.get_ptr(), wh, drag_pixel-pos_diff, drag_pixel+pos_diff, dragger_opacity + 0x00ffffff);

    } else if (state["dragger_type"] == 2 && state["dragger_shown"] != 0){
        ivec2 pos_diff_0((int) (dragger_size),(int) (dragger_size*0.4));
        ivec2 pos_diff_1((int) (dragger_size*0.4),(int) (dragger_size));
        draw_quadrilateral(gpu_pix.get_ptr(), wh, 
            drag_pixel-pos_diff_0, drag_pixel-pos_diff_1, drag_pixel+pos_diff_0, drag_pixel+pos_diff_1, 
            dragger_opacity + bg_color);
        pos_diff_0 = pos_diff_0*ivec2(-1,1);
        pos_diff_1 = pos_diff_1*ivec2(-1,1);
        draw_quadrilateral(gpu_pix.get_ptr(), wh, 
            drag_pixel-pos_diff_0, drag_pixel-pos_diff_1,drag_pixel+pos_diff_0, drag_pixel+pos_diff_1, 
            dragger_opacity + bg_color);

        pos_diff_0 = ivec2((int) (dragger_size*0.7),(int) (dragger_size*0.4));
        pos_diff_1 = ivec2((int) (dragger_size*0.4),(int) (dragger_size*0.7));
        draw_quadrilateral(gpu_pix.get_ptr(), wh, 
            drag_pixel-pos_diff_0, drag_pixel-pos_diff_1, drag_pixel+pos_diff_0, drag_pixel+pos_diff_1, 
            dragger_opacity + 0x00ffffff);
        pos_diff_0 = pos_diff_0*ivec2(-1,1);
        pos_diff_1 = pos_diff_1*ivec2(-1,1);
        draw_quadrilateral(gpu_pix.get_ptr(), wh, 
            drag_pixel-pos_diff_0, drag_pixel-pos_diff_1, drag_pixel+pos_diff_0, drag_pixel+pos_diff_1, 
            dragger_opacity + 0x00ffffff);
    }






    int diagram_unit = get_diagram_unit(wh,state["top_y"],state["bottom_y"],state["diagram_opacity"]);
    float point_radius = diagram_unit*0.18;
    ivec2 diagram_origin = get_diagram_origin(wh, state["diagram_opacity"], diagram_unit);
    ivec2 diagram_label(0,diagram_unit*0.7*state["diagram_label"]);
    const vec2 textbox_size(point_radius * 2.8);
    const vec2 textbox_offset = vec2(0,point_radius*0.1);

    // const int opacity = ((int) state["diagram_opacity"]) << 24;
    const int opacity = max(0,(int) state["diagram_opacity"]*2-255) << 24;
    if (opacity != 0){

        const int axis_color = ((int) (state["bg_1_r"]*1.5+40)) << 16 | ((int) (state["bg_1_g"]*1.5+40)) << 8 | (int) (state["bg_1_b"]*1.5+40);
        int axis_width = wh.y*0.004;

        draw_rectangle(gpu_pix.get_ptr(), wh, 
            ivec2(0,0), diagram_origin*2+diagram_label, opacity + bg_color);
        draw_rectangle(gpu_pix.get_ptr(), wh, 
            diagram_origin-ivec2(axis_width,diagram_unit)+diagram_label, diagram_origin+ivec2(axis_width,diagram_unit)+diagram_label, opacity + axis_color);
        draw_rectangle(gpu_pix.get_ptr(), wh, 
            diagram_origin-ivec2(diagram_unit,axis_width)+diagram_label, diagram_origin+ivec2(diagram_unit,axis_width)+diagram_label, opacity + axis_color);
        
        
        const int border_opacity = max(0,(int) state["diagram_opacity"]*11-2550) << 24;
        draw_rectangle(gpu_pix.get_ptr(), wh, 
            ivec2(diagram_origin.x*2-axis_width,0), diagram_origin*2+axis_width+diagram_label, border_opacity + axis_color);
        draw_rectangle(gpu_pix.get_ptr(), wh, 
            ivec2(0,diagram_origin.y*2-axis_width)+diagram_label, diagram_origin*2+axis_width+diagram_label, border_opacity + axis_color);
    }




    const int xx_opacity = ((int) state["xx_opacity"]) << 24;
    if (xx_opacity != 0){
        const vec2 xx_pos = vec2(state["xx_x"], -state["xx_y"])*diagram_unit+diagram_origin+diagram_label;
        draw_circle(gpu_pix.get_ptr(), wh, xx_pos, point_radius*1.1, xx_opacity + bg_color,1.0);
        draw_circle(gpu_pix.get_ptr(), wh, xx_pos, point_radius, xx_opacity + 0x00dd44dd,1.0);
        set_global_state("xx_pos_x", xx_pos.x/wh.x);
        set_global_state("xx_pos_y", xx_pos.y/wh.y);
        write_text(gpu_pix.get_ptr(), wh, latex_color(bg_color, "x^2"), xx_pos+textbox_offset*0.5, textbox_size, state["xx_opacity"]/255*state["x_label"], 0);
        write_text(gpu_pix.get_ptr(), wh, latex_color(bg_color, "1"), xx_pos+textbox_offset*0.5, textbox_size, state["xx_opacity"]/255*(1-state["x_label"]), 0);
    }

    const int xy_opacity = ((int) state["xy_opacity"]) << 24;
    if (xy_opacity != 0){
        const vec2 xy_pos = vec2(state["xy_x"], -state["xy_y"])*diagram_unit+diagram_origin+diagram_label;
        draw_circle(gpu_pix.get_ptr(), wh, xy_pos, point_radius*1.1, xy_opacity + bg_color,1.0);
        draw_circle(gpu_pix.get_ptr(), wh, xy_pos, point_radius, xy_opacity + 0x00ccccee,1.0);
        set_global_state("xy_pos_x", xy_pos.x/wh.x);
        set_global_state("xy_pos_y", xy_pos.y/wh.y);
        write_text(gpu_pix.get_ptr(), wh, latex_color(bg_color, "xy"), xy_pos+textbox_offset, textbox_size, state["xy_opacity"]/255*state["x_label"], 0);
        write_text(gpu_pix.get_ptr(), wh, latex_color(bg_color, "y"), xy_pos+textbox_offset, textbox_size, state["xy_opacity"]/255*(1-state["x_label"]), 0);
    }
    
    
    const int yy_opacity = ((int) state["yy_opacity"]) << 24;
    if (yy_opacity != 0){
        const vec2 yy_pos = vec2(state["yy_x"], -state["yy_y"])*diagram_unit+diagram_origin+diagram_label;
        const vec2 yy_transition(0,point_radius*2);
        draw_circle(gpu_pix.get_ptr(), wh, yy_pos, point_radius*1.1, yy_opacity + bg_color,1.0);
        draw_circle(gpu_pix.get_ptr(), wh, yy_pos, point_radius, yy_opacity + 0x00dddd44,1.0);
        set_global_state("yy_pos_x", yy_pos.x/wh.x);
        set_global_state("yy_pos_y", yy_pos.y/wh.y);
        write_text(gpu_pix.get_ptr(), wh, latex_color(bg_color, "y^2"), yy_pos+textbox_offset+yy_transition*(1-state["y_label"]), 
            textbox_size, state["yy_opacity"]/255*state["y_label"], 0);
        write_text(gpu_pix.get_ptr(), wh, latex_color(bg_color, "i^2"), yy_pos-yy_transition*state["y_label"], 
            textbox_size*0.9, state["yy_opacity"]/255*(1-state["y_label"]), 0);
    }




    const ivec2 point_dist(state["point_x"]*screen_unit,-state["point_y"]*screen_unit);
    for  (int p = 0; p < state["point_count"]; p++){

        const float point_lerp = min(state["point_count"]-p,1.0);
        const int point_opacity = ((int) state["point_opacity"]) << 24;
        const int point_size = point_radius*0.85*sin(point_lerp*1.98)*state["point_size"];

        const int point_stroke = 0x00ffffff;//(state["point_label"] != 0) ? 0x00ffff00 : 0x00ffffff;

        const vec2 point_loc = origin+(p+1)*point_dist;
        draw_circle(gpu_pix.get_ptr(), wh, point_loc, point_size, point_opacity+point_stroke,1.0);
        draw_circle(gpu_pix.get_ptr(), wh, point_loc, point_size*0.85, point_opacity+bg_color,1.0);
        if (state["point_label"] != 0 && point_size > 1){
            const string suffix = (p == 3) ? "^t^h" : "";
            write_text(gpu_pix.get_ptr(), wh, latex_color(point_stroke,to_string(p+1)+suffix), point_loc, point_size*2, state["point_opacity"]/255, 0);
        }

    }


}


