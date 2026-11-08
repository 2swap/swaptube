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

struct Interpolation {
    ivec2 wh_1;
    uint32_t* pix_1;

    ivec2 wh_2;
    uint32_t* pix_2;

    int num_glyphs_1;
    Glyph* glyphs_1;

    int num_glyphs_2;
    Glyph* glyphs_2;

    // The chosen matching. match_1[i] is the index of the image 2 glyph that image 1 glyph i moves into,
    // or -1 if it fades out. match_2 is the inverse: the image 1 glyph that moves into image 2 glyph j, or -1 if it fades in.
    int* match_1;
    int* match_2;
};

SHARED_FILE_SUFFIX
