#include "ConwayScene.h"
#include "../../Host_Device_Shared/find_roots.c"

extern "C" void draw_conway(
    Bitboard* h_board, Bitboard* h_board_2,
    const ivec2& grid_wh_bitboards,
    uint32_t* h_pixels, const ivec2& pix_wh,
    const vec2& lx_ty, const vec2& rx_by, float transition
);

ConwayScene::ConwayScene(const ivec2& size_bitboards, const vec2& dimensions) : CoordinateScene(dimensions) {
    conway_grid = new ConwayGrid(size_bitboards);
    add_data_object(conway_grid);
}

void ConwayScene::draw() {
        conway_grid->iterate();
    draw_conway(
        conway_grid->d_board_2,
        conway_grid->d_board,
        conway_grid->grid_wh_bitboards,
        gpu_pix->get_ptr(), get_width_height(),
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
        ;//conway_grid->iterate();
}
