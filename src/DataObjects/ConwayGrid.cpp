#include "ConwayGrid.h"
#include <vector>

extern "C" void iterate_conway(Bitboard*& d_board, Bitboard*& d_board_2, const ivec2& wh_bitboards, const int iterations);
extern "C" void allocate_conway_grid(Bitboard** d_board, const ivec2& grid_wh_bitboards, const uint32_t* envelope, const ivec2& envelope_wh);
extern "C" void free_conway_grid(Bitboard* d_board);

ConwayGrid::ConwayGrid(const ivec2& wh_bitboards, const Pixels& env) : grid_wh_bitboards(wh_bitboards) {
    allocate_conway_grid(&d_board, wh_bitboards, env.pixels.data(), env.wh);
    allocate_conway_grid(&d_board_2, wh_bitboards, env.pixels.data(), env.wh);
}
ConwayGrid::~ConwayGrid() {
    free_conway_grid(d_board);
    free_conway_grid(d_board_2);
}
void ConwayGrid::iterate() {
    iterate_conway(d_board, d_board_2, grid_wh_bitboards, 1);
}
