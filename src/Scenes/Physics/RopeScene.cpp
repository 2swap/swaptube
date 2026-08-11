#include "RopeScene.h"
#include <iostream>

extern "C" void cuda_render_rope(uint32_t* pixels, const ivec2& wh, const vec2* rope, const int rope_length,
     const vec2& lx_ty, const vec2& rx_by);
extern "C" void copy_pins(const vec2* h_pins, vec2* d_pins, const int pins_length);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);



RopeScene::RopeScene(const string file_name, const vec2& dimensions) : rope(file_name) {
    manager.set({
        {"center_x", "0.5"},
        {"center_y", "0.5"},
        {"zoom", "1.5"},
    });
}

void RopeScene::add_pin(vec2 pos){
    rope.add_pin(pos);
}

void RopeScene::remove_pin(int pin_index){
    rope.remove_pin(pin_index);
}

void RopeScene::draw(){
    cuda_render_rope(gpu_pix.get_ptr(), get_width_height(), rope.d_nodes, 1000, 
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]));
    for (const auto& pin : rope.h_pins) {
        draw_circle(gpu_pix.get_ptr(), get_width_height(), point_to_pixel(pin), 5, 0xFFFF0000);
    }
}

void RopeScene::set_pins(vec2 pos, uint32_t color, float size){
}

void RopeScene::change_data() {
    CoordinateScene::change_data();
    rope.tick();
}
