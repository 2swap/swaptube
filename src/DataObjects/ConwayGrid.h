#pragma once

#include <cstdint>
#include "DataObject.h"

typedef uint64_t Bitboard;

class ConwayGrid : public DataObject {
public:
    ivec2 grid_wh_bitboards;
    Bitboard* d_board;
    Bitboard* d_board_2;
    ConwayGrid(const ivec2& wh_bitboards);
    ~ConwayGrid();
    void tick(const StateReturn& state);
    void iterate();
};
