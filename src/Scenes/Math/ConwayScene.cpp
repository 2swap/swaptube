#include "ConwayScene.h"
#include "../../Host_Device_Shared/find_roots.c"

extern "C" void draw_conway(
    Bitboard* h_board, Bitboard* h_board_2,
    const vec2& size_bitboards,
    unsigned int* h_pixels, const vec2& pix_size,
    const vec2& lx_ty, const vec2& rx_by, float transition
);

ConwayScene::ConwayScene(const vec2& dimensions) : CoordinateScene(dimensions), conway_grid(size_bitboards * 8) {
    conway_grid.iterate();
}

void ConwayScene::draw() {
    draw_conway(
        conway_grid.d_board_2,
        conway_grid.d_board,
        size_bitboards,
        pix.pixels.data(), pix.size,
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
        conway_grid.iterate();
}

void ConwayScene::change_data() {
    //conway_grid.iterate();
}

bool ConwayScene::check_if_data_changed() const {
    return conway_grid.has_been_updated_since_last_scene_query();
}
