#include "RopeScene.h"
#include <iostream>

extern "C" void cuda_render_path(uint32_t* pixels, const ivec2& wh, const vec2* rope, const int rope_length,
     const vec2& lx_ty, const vec2& rx_by, const uint32_t color, const float opacity, const float thickness, const bool closed);
extern "C" void copy_pins(const vec2* h_pins, vec2* d_pins, const int pins_length);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);



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

void RopeScene::change_data() {
    CoordinateScene::change_data();
    rope.tick();

    if (progress < 1.0f) {
        progress += draw_speed;
        if (progress > 1.0f) progress = 1.0f;
    }
}

void RopeScene::draw() {
    const int total_nodes = 1000;
    int visible_nodes = static_cast<int>(total_nodes * progress);

    if (visible_nodes > 1) {
        bool is_closed = (progress >= 1.0f); 

        cuda_render_path(gpu_pix.get_ptr(), get_width_height(), rope.d_nodes, visible_nodes,
            vec2(state["left_x"], state["top_y"]),
            vec2(state["right_x"], state["bottom_y"]),
            0xFFFFFFFF, 1.0f, 2.0f, is_closed);
    }

    for (const auto& pin : rope.h_pins) {
        draw_circle(gpu_pix.get_ptr(), get_width_height(), point_to_pixel(pin), 5, 0xFFFF0000, 1.0f);
    }
}

void RopeScene::set_pins(vec2 pos, uint32_t color, float size){
}

