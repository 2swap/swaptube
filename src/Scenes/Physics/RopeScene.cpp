#include "RopeScene.h"
#include <iostream>

extern "C" void cuda_render_rope(uint32_t* pixels, const ivec2& wh, const vec2* rope, const int rope_length, const vec2* pins, const int pins_length,
     const vec2& lx_ty, const vec2& rx_by);


RopeScene::RopeScene(const vec2& dimensions){
    rope = new Rope("io_in/loop_exemple_0");
    add_data_object(rope);
    manager.set({
        {"center_x", "0.5"},
        {"center_y", "0.5"},
        {"zoom", "1.5"},
    });
    
}

void RopeScene::draw(){
    cout << "RopeScene::draw() called" << endl;
    cuda_render_rope(gpu_pix->get_ptr(), get_width_height(), rope->d_nodes, 1000, rope->d_pins, 10, 
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]));
}


const StateQuery RopeScene::populate_state_query() const{
    return CoordinateScene::populate_state_query();
}



void RopeScene::set_pins(vec2 pos, uint32_t color, float size){
}