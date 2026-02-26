#include "ConwayGrid.h"
#include <vector>

extern "C" void iterate_conway(Bitboard* d_board, Bitboard* d_board_2, int w_bitboards, int h_bitboards);
extern "C" void allocate_conway_grid(Bitboard** d_board, Bitboard** d_board_2, int w_bitboards, int h_bitboards);
extern "C" void free_conway_grid(Bitboard* d_board, Bitboard* d_board_2);

ConwayGrid::ConwayGrid(const int width, const int height) : w_bitboards(width/8), h_bitboards(height/8) {
    mark_updated();
    allocate_conway_grid(&d_board, &d_board_2, w_bitboards, h_bitboards);
}
ConwayGrid::~ConwayGrid() {
    mark_updated();
    free_conway_grid(d_board, d_board_2);
}
void ConwayGrid::iterate() {
    mark_updated();
    iterate_conway(d_board, d_board_2, w_bitboards, h_bitboards);
    Bitboard* temp = d_board;
    d_board = d_board_2;
    d_board_2 = temp;
}
