
#include "CursorScene.h"

extern "C" void draw_quadrilateral(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const vec2& p3, const uint32_t color);
extern "C" void draw_triangle(uint32_t* pix, const ivec2& wh, const vec2& p0, const vec2& p1, const vec2& p2, const uint32_t color);


CursorScene::CursorScene(const vec2& dimensions) : Scene(dimensions) {
    manager.set({
        {"cursor_x", "0.5"},{"cursor_y", "0.5"},{"cursor_size", "0.1"},
        {"fill_r", "255"},{"fill_g", "255"},{"fill_b", "255"},
        {"stroke_r", "0"},{"stroke_g", "0"},{"stroke_b", "68"},
    });
}

void CursorScene::draw() {

    const ivec2 wh = get_width_height();

    const int fill_color = 0xff000000 | ((int) state["fill_r"]) << 16 | ((int) state["fill_g"]) << 8 | (int) state["fill_b"];
    const int stroke_color = 0xff000000 | ((int) state["stroke_r"]) << 16 | ((int) state["stroke_g"]) << 8 | (int) state["stroke_b"];

    const ivec2 cursor_pos = ivec2((int) (state["cursor_x"]*wh.x),(int) (state["cursor_y"]*wh.y));
    const float cursor_size = state["cursor_size"]*wh.y;


    draw_triangle(gpu_pix.get_ptr(), wh, 
        cursor_pos+ivec2((int) (cursor_size*-0.3),(int) (cursor_size*-0.51)), 
        cursor_pos+ivec2((int) (cursor_size*-0.3),(int) (cursor_size*0.46)), 
        cursor_pos+ivec2((int) (cursor_size*0.19),0), 
        stroke_color);
    draw_triangle(gpu_pix.get_ptr(), wh, 
        cursor_pos+ivec2((int) (cursor_size*-0.3),(int) (cursor_size*-0.51)), 
        cursor_pos+ivec2((int) (cursor_size*0.43),(int) (cursor_size*0.15)), 
        cursor_pos+ivec2((int) (cursor_size*-0.1),(int) (cursor_size*0.15)), 
        stroke_color);
    draw_quadrilateral(gpu_pix.get_ptr(), wh, 
        cursor_pos+ivec2((int) (cursor_size*-0.03),(int) (cursor_size*-0.13)), 
        cursor_pos+ivec2((int) (cursor_size*-0.2),(int) (cursor_size*-0.07)), 
        cursor_pos+ivec2((int) (cursor_size*0.02),(int) (cursor_size*0.44)), 
        cursor_pos+ivec2((int) (cursor_size*0.21),(int) (cursor_size*0.35)), 
        stroke_color);


    draw_triangle(gpu_pix.get_ptr(), wh, 
        cursor_pos+ivec2((int) (cursor_size*-0.25),(int) (cursor_size*-0.4)), 
        cursor_pos+ivec2((int) (cursor_size*-0.25),(int) (cursor_size*0.35)), 
        cursor_pos+ivec2((int) (cursor_size*0.10),0), 
        fill_color);
    draw_triangle(gpu_pix.get_ptr(), wh, 
        cursor_pos+ivec2((int) (cursor_size*-0.25),(int) (cursor_size*-0.40)), 
        cursor_pos+ivec2((int) (cursor_size*0.30),(int) (cursor_size*0.1)), 
        cursor_pos+ivec2((int) (cursor_size*-0.1),(int) (cursor_size*0.1)), 
        fill_color);
    draw_quadrilateral(gpu_pix.get_ptr(), wh, 
        cursor_pos+ivec2((int) (cursor_size*-0.07),(int) (cursor_size*-0.12)), 
        cursor_pos+ivec2((int) (cursor_size*-0.15),(int) (cursor_size*-0.08)), 
        cursor_pos+ivec2((int) (cursor_size*0.05),(int) (cursor_size*0.36)), 
        cursor_pos+ivec2((int) (cursor_size*0.13),(int) (cursor_size*0.32)), 
        fill_color);

}


