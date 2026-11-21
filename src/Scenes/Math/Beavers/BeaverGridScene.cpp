#include "BeaverGridScene.h"

extern "C" void beaver_grid_cuda(int num_states, unsigned int* pixels, const ivec2& wh, const vec2& lx_ty, const vec2& rx_by, int max_steps);

BeaverGridScene::BeaverGridScene(const int num_states, const vec2& dimension)
: CoordinateScene(dimension) {
    manager.set("max_steps", "0");
    manager.set("ticks_opacity", "1");
    manager.set("num_states", std::to_string(num_states));
}

void BeaverGridScene::draw() {
    beaver_grid_cuda(
        state["num_states"],
        gpu_pix.get_ptr(), get_width_height(),
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]),
        state["max_steps"]
    );

    CoordinateScene::draw();
}
