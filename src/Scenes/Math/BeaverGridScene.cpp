#include "BeaverGridScene.h"

extern "C" void beaver_grid_cuda(int num_states, int num_symbols, unsigned int* pixels, vec2 size, vec2 lx_ty, vec2 rx_by, int max_steps);

BeaverGridScene::BeaverGridScene(const int num_states, const int num_symbols, const vec2& dimension)
: CoordinateScene(dimension) {
    manager.set("max_steps", "0");
    manager.set("ticks_opacity", "1");
    manager.set("num_states", std::to_string(num_states));
    manager.set("num_symbols", std::to_string(num_symbols));
}

void BeaverGridScene::draw() {
    beaver_grid_cuda(
        state["num_states"], state["num_symbols"],
        pix.pixels.data(), pix.size,
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]),
        state["max_steps"]
    );

    CoordinateScene::draw();
}

const StateQuery BeaverGridScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, { "max_steps", "num_states", "num_symbols" });
    return sq;
}

void BeaverGridScene::mark_data_unchanged() { }
void BeaverGridScene::change_data() { }
bool BeaverGridScene::check_if_data_changed() const { return false; }
