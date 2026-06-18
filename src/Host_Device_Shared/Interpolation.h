#pragma once

#include <cstdint>
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

// Struct to represent a glyph, which is a connected component of pixels in the image.
struct Glyph {
    ivec2 wh;
    ivec2 top_left;
    uint32_t* pix;
};

// Enum for the status of an edge in the adjacency graph.
enum EdgeStatus {
    NO_EDGE = 0, // Either glyphs are different, or we discarded the edge.
    KEEP_EDGE = 1,
    UNDECIDED_EDGE = 2
};

struct GraphAdjacency {
    EdgeStatus status;
    vec2 position_delta;
};

struct GraphAdjacencyMatrix {
    // Adjacency matrix of the graph, where rows represent components in image 1, and columns represent components in image 2.
    GraphAdjacency* adj_matrix;
    uint32_t num_components_1;
    uint32_t num_components_2;
};

struct Interpolation {
    ivec2 wh_1;
    uint32_t* pix_1;

    ivec2 wh_2;
    uint32_t* pix_2;

    int num_glyphs_1;
    Glyph* glyphs_1;

    int num_glyphs_2;
    Glyph* glyphs_2;

    GraphAdjacencyMatrix adjacency_matrix;
};

SHARED_FILE_SUFFIX
