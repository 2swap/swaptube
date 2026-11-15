#include "Smoketest.h"
#include "Pixels.h"

// Segments a rendered LaTeX image using its glyph ID image (see latex_to_glyph_ids in IO/Latex.h).
// element_of_label[label] is the SVG element ID each glyph came from (0 if unknown); index 0 is unused.
Pixels segment_by_glyph_ids(const Pixels& visible, const Pixels& ids, unsigned int& count, vector<uint32_t>& element_of_label);

Pixels colorize_segments(const Pixels& segmented);

