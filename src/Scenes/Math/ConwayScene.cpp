#include "ConwayScene.h"
#include "../../Host_Device_Shared/find_roots.c"

extern "C" void draw_conway(
    Bitboard* h_board, Bitboard* h_board_2,
    int w_bitboards, int h_bitboards,
    unsigned int* h_pixels, int pixels_w, int pixels_h,
    vec2 lx_ty, vec2 rx_by, float transition
);

ConwayScene::ConwayScene(const vec2& dimensions) : CoordinateScene(dimensions), conway_grid(grid_width * 8, grid_height * 8) { }

void ConwayScene::draw() {
    draw_conway(
        conway_grid.d_board_2,
        conway_grid.d_board,
        grid_width,
        grid_height,
        pix.pixels.data(), pix.w, pix.h,
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]),
        state["microblock_fraction_passthrough"]
    );
    CoordinateScene::draw();
}

const StateQuery ConwayScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {
        "microblock_fraction_passthrough"
    });
    return sq;
}

void ConwayScene::on_end_transition_extra_behavior(const TransitionType tt){
    if(tt == MICRO)
        ;//conway_grid.iterate();
}
