#include "Smoketest.h"
#include "Pixels.h"

void flood_fill(Pixels& ret, const Pixels& p, int start_x, int start_y, int color);

Pixels segment(const Pixels& p, unsigned int& id);

Pixels colorize_segments(const Pixels& segmented);

