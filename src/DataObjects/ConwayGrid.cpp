#include "ConwayGrid.h"
#include <vector>

extern "C" void iterate_conway(Bitboard*& d_board, Bitboard*& d_board_2, const ivec2& wh_bitboards, const int iterations);
extern "C" void reverse_conway_loop(Bitboard*& d_board, Bitboard*& d_board_2, Bitboard*& target, const ivec2& grid_wh_bitboards, int max_iters);
extern "C" void allocate_conway_grid(Bitboard** d_board, const ivec2& grid_wh_bitboards, const uint32_t* envelope, const ivec2& envelope_wh);
extern "C" void free_conway_grid(Bitboard* d_board);

ConwayGrid::ConwayGrid(const ivec2& wh_bitboards, const Pixels& env) : grid_wh_bitboards(wh_bitboards) {
    mark_updated();
    allocate_conway_grid(&d_board, wh_bitboards, env.pixels.data(), env.wh);
    allocate_conway_grid(&d_board_2, wh_bitboards, env.pixels.data(), env.wh);
    //allocate_conway_grid(&d_target, wh_bitboards, env.pixels.data(), env.wh);
}
ConwayGrid::~ConwayGrid() {
    mark_updated();
    free_conway_grid(d_board);
    free_conway_grid(d_board_2);
    //free_conway_grid(d_target);
}
void ConwayGrid::tick(const StateReturn& state) {}
void ConwayGrid::iterate() {
    mark_updated();
    iterate_conway(d_board, d_board_2, grid_wh_bitboards, 1);
    //reverse_conway_loop(d_board, d_board_2, d_target, grid_wh_bitboards, 1);
}
