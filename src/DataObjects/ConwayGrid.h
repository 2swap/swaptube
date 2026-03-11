#pragma once

#include <cstdint>
#include "DataObject.h"
#include "../Host_Device_Shared/vec.h"

typedef uint64_t Bitboard;

class ConwayGrid : public DataObject {
public:
    const vec2 size_bitboards;
    Bitboard* d_board;
    Bitboard* d_board_2;
    ConwayGrid(const vec2& sz);
    ~ConwayGrid();
    void iterate();
};
