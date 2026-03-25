#include "BeaverGridTNFScene.h"

extern "C" void beaver_grid_TNF_cuda(unsigned int* pixels, int w, int h, vec2 lx_ty, vec2 rx_by, int max_steps);

BeaverGridTNFScene::BeaverGridTNFScene(const vec2& dimension)
: CoordinateScene(dimension) {
    manager.set("max_steps", "0");
    manager.set("ticks_opacity", "1");
}

void BeaverGridTNFScene::draw() {
    beaver_grid_TNF_cuda(
        pix.pixels.data(), pix.w, pix.h,
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]),
        state["max_steps"]
    );

    CoordinateScene::draw();
}

const StateQuery BeaverGridTNFScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    sq.insert("max_steps");
    return sq;
}

void BeaverGridTNFScene::mark_data_unchanged() { }
void BeaverGridTNFScene::change_data() { }
bool BeaverGridTNFScene::check_if_data_changed() const { return false; }
