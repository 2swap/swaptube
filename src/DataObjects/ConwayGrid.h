#pragma once

#include <cstdint>
#include "../Core/Pixels.h"

typedef uint64_t Bitboard;

class ConwayGrid {
public:
    ivec2 grid_wh_bitboards;
    Bitboard* d_board;
    Bitboard* d_board_2;
    ConwayGrid(const ivec2& wh_bitboards, const Pixels& env);
    ~ConwayGrid();
    void iterate();
};
