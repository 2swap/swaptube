#include <cstdint>
#include "Convolution.h"

void flood_fill(Pixels& ret, const Pixels& p, int start_x, int start_y, int color) {
    stack<pair<int, int>> stack;
    stack.push({start_x, start_y});

    while (!stack.empty()) {
        auto [x, y] = stack.top();
        stack.pop();

        if (p.out_of_range(x, y) || p.get_alpha(x, y) < 128 || ret.get_pixel_carelessly(x, y) != 0)
            continue;

        ret.set_pixel_carelessly(x, y, color);

        stack.push({x + 1, y}); // Right
        stack.push({x - 1, y}); // Left
        stack.push({x, y + 1}); // Down
        stack.push({x, y - 1}); // Up
    }
}

Pixels segment(const Pixels& p, uint32_t& id) {
    Pixels ret(p.wh);

    id = 0u;

    // Perform flood fill for each pixel in the input TranslatedPixels
    for (int y = 0; y < ret.wh.y; y++) {
        for (int x = 0; x < ret.wh.x; x++) {
            if (p.get_alpha(x, y) >= 128 && ret.get_pixel_carelessly(x, y) == 0) {
                id++;
                flood_fill(ret, p, x, y, id);
            }
        }
    }

    colorize_segments(ret).print_to_terminal();
    return ret;
}

Pixels colorize_segments(const Pixels& segmented) {
    Pixels colorized(segmented.wh);

    // A simple function to generate a deterministic but pseudo-random color based on segment ID
    auto segment_id_to_color = [](uint32_t id) -> int {
        // Example: Use a hash-like function to generate color from segment ID
        uint32_t r = (id * 137 + 113) % 256;
        uint32_t g = (id * 149 + 157) % 256;
        uint32_t b = (id * 163 + 173) % 256;
        return (r << 16) | (g << 8) | b;
    };

    // Iterate over each pixel in the segmented image
    for (int y = 0; y < segmented.wh.y; ++y) {
        for (int x = 0; x < segmented.wh.x; ++x) {
            colorized.set_pixel_carelessly(x, y, segment_id_to_color(segmented.get_pixel_carelessly(x, y)) | 0xff000000);  // Copy the original pixel if not part of any segment
        }
    }

    return colorized;
}

