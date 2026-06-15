#pragma once

#include <cstdint>
#include "DataObject.h"
#include "../Core/Pixels.h"

typedef uint64_t Bitboard;

class ConwayGrid : public DataObject {
public:
    ivec2 grid_wh_bitboards;
    Bitboard* d_board;
    Bitboard* d_board_2;
    Bitboard* d_target;
    ConwayGrid(const ivec2& wh_bitboards, const Pixels& env);
    ~ConwayGrid();
    void tick(const StateReturn& state);
    void iterate();
};
