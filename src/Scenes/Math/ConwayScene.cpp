#include "ConwayScene.h"
#include "../../Host_Device_Shared/find_roots.c"

extern "C" void draw_conway(
    Bitboard* h_board, Bitboard* h_board_2,
    const ivec2& grid_wh_bitboards,
    uint32_t* h_pixels, const ivec2& pix_wh,
    const vec2& lx_ty, const vec2& rx_by, float transition
);

ConwayScene::ConwayScene(const vec2& dimensions) : CoordinateScene(dimensions), conway_grid(ivec2(10000, 10000)), d_pixels(get_pixels_size()) {
    add_data_object(&conway_grid);
}

void ConwayScene::draw() {
    draw_conway(
        conway_grid.d_board_2,
        conway_grid.d_board,
        conway_grid.grid_wh_bitboards,
        d_pixels.get_ptr(), pix.wh,
        vec2(state[ "left_x"], state[   "top_y"]),
        vec2(state["right_x"], state["bottom_y"]),
        state["microblock_fraction_passthrough"]
    );
    d_pixels.copy_to_host(pix.pixels.data(), pix.wh);
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
        conway_grid.iterate();
}
