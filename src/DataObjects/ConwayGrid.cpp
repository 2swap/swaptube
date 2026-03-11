#include "ConwayGrid.h"
#include <vector>

extern "C" void iterate_conway(Bitboard* d_board, Bitboard* d_board_2, const vec2& size_bitboards);
extern "C" void allocate_conway_grid(Bitboard** d_board, Bitboard** d_board_2, const vec2& size_bitboards);
extern "C" void free_conway_grid(Bitboard* d_board, Bitboard* d_board_2);

ConwayGrid::ConwayGrid(const vec2& sz) : size_bitboards(sz / 8) {
    mark_updated();
    allocate_conway_grid(&d_board, &d_board_2, size_bitboards);
}
ConwayGrid::~ConwayGrid() {
    mark_updated();
    free_conway_grid(d_board, d_board_2);
}
void ConwayGrid::iterate() {
    mark_updated();
    iterate_conway(d_board, d_board_2, size_bitboards);
    Bitboard* temp = d_board;
    d_board = d_board_2;
    d_board_2 = temp;
}
