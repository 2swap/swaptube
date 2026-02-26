#pragma once

#include <cstdint>
#include "DataObject.h"

typedef uint64_t Bitboard;

class ConwayGrid : public DataObject {
public:
    int w_bitboards; int h_bitboards;
    Bitboard* d_board;
    Bitboard* d_board_2;
    ConwayGrid(const int width, const int height);
    ~ConwayGrid();
    void iterate();
};
