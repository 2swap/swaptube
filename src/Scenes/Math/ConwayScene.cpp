#include "../Common/CoordinateScene.h"
#include "../../Host_Device_Shared/find_roots.c"
#include "../../DataObjects/ConwayGrid.h"
#include "../../Core/State/StateManager.h"

extern "C" void draw_conway(
    Bitboard* h_board, Bitboard* h_board_2,
    int w_bitboards, int h_bitboards,
    unsigned int* h_pixels, int pixels_w, int pixels_h,
    vec2 lx_ty, vec2 rx_by, float transition
);

class ConwayScene : public CoordinateScene {
private:
    // 10 by 10 grid of conway cells
    int grid_width = 10000;
    int grid_height = 10000;
    ConwayGrid conway_grid;

public:
    ConwayScene(const float width = 1, const float height = 1) : CoordinateScene(width, height), conway_grid(grid_width * 8, grid_height * 8) {
        conway_grid.iterate();
    }

    void draw() override {
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

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        state_query_insert_multiple(sq, {
            "microblock_fraction_passthrough"
        });
        return sq;
    }

    void on_end_transition_extra_behavior(const TransitionType tt){
        if(tt == MICRO)
        conway_grid.iterate();
    }

    void change_data() override {
        //conway_grid.iterate();
    }

    bool check_if_data_changed() const override {
        return conway_grid.has_been_updated_since_last_scene_query();
    }
};
