#include "ConwayGrid.h"
#include "../Core/Pixels.h"
#include "../IO/SVG.h"
#include <vector>

extern "C" void iterate_conway(Bitboard*& d_board, Bitboard*& d_board_2, const ivec2& wh_bitboards, const int iterations);
extern "C" void allocate_conway_grid(Bitboard** d_board, Bitboard** d_board_2, const ivec2& grid_wh_bitboards);
extern "C" void free_conway_grid(Bitboard* d_board, Bitboard* d_board_2);

ConwayGrid::ConwayGrid(const ivec2& wh_bitboards) : grid_wh_bitboards(wh_bitboards) {
    mark_updated();
    allocate_conway_grid(&d_board, &d_board_2, wh_bitboards);
}
ConwayGrid::~ConwayGrid() {
    mark_updated();
    free_conway_grid(d_board, d_board_2);
}
void ConwayGrid::tick(const StateReturn& state) {}
void ConwayGrid::iterate() {
    mark_updated();
    iterate_conway(d_board, d_board_2, grid_wh_bitboards, 1);
}
