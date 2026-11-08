#include <cstdint>
#include <unordered_map>
#include "GlyphSegmentation.h"

// Segments using a glyph ID image (see latex_to_glyph_ids) instead of pure connectivity, so that
// multi-part symbols (the dot and stem of an i, the two bars of =) stay one glyph.
// - "On" pixels are all visible ones (alpha > 0), so a glyph keeps its antialiased edges, and a small glyph
//   whose thin strokes are all partially covered isn't lost.
// - Each on pixel takes the element ID under it. Antialiased fringe pixels that the crisp ID render
//   missed take the ID of the nearest labeled on pixel. As a safety net, a region the ID render missed
//   entirely (e.g. a sub-pixel hairline) becomes its own glyph rather than vanishing during transitions.
// - Each SVG element is its own glyph. Elements are never merged based on their pixels, since pixels touch or
//   not depending on resolution; what a glyph is comes from the SVG (see latex_glyph_shapes).
// element_of_label[label] is the element ID that glyph `label` came from, or 0 for a fallback region.
Pixels segment_by_glyph_ids(const Pixels& visible, const Pixels& ids, uint32_t& count, vector<uint32_t>& element_of_label) {
    if (visible.wh.x != ids.wh.x || visible.wh.y != ids.wh.y)
        throw runtime_error("segment_by_glyph_ids: visible and ID images differ in size");
    const int W = visible.wh.x, H = visible.wh.y;
    const int N = W * H;
    const uint32_t FALLBACK_BASE = 0x1000000; // IDs for regions the ID image didn't cover (above any element ID)

    auto on = [&](int p) { return visible.get_alpha(p % W, p / W) > 0; };
    vector<uint32_t> elem(N, 0);

    // 1. Read element IDs under on pixels, and seed a BFS from them.
    vector<int> frontier;
    for (int p = 0; p < N; p++) {
        if (!on(p)) continue;
        const int x = p % W, y = p / W;
        if (ids.get_alpha(x, y) >= 128) {
            elem[p] = ids.get_pixel_carelessly(x, y) & 0xFFFFFF;
            if (elem[p] != 0) frontier.push_back(p);
        }
    }

    // 2. Spread IDs into unlabeled on pixels (antialiasing fringe), nearest first.
    const int dx4[4] = {1, -1, 0, 0}, dy4[4] = {0, 0, 1, -1};
    for (size_t head = 0; head < frontier.size(); head++) {
        const int p = frontier[head], x = p % W, y = p / W;
        for (int k = 0; k < 4; k++) {
            const int nx = x + dx4[k], ny = y + dy4[k];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int q = ny * W + nx;
            if (elem[q] == 0 && on(q)) { elem[q] = elem[p]; frontier.push_back(q); }
        }
    }

    // 3. Anything still unlabeled (an element the crisp render missed entirely) gets one ID per connected region.
    uint32_t next_fallback = FALLBACK_BASE;
    for (int p = 0; p < N; p++) {
        if (elem[p] != 0 || !on(p)) continue;
        const uint32_t fid = next_fallback++;
        vector<int> st = {p};
        elem[p] = fid;
        while (!st.empty()) {
            const int c = st.back(); st.pop_back();
            const int x = c % W, y = c / W;
            for (int k = 0; k < 4; k++) {
                const int nx = x + dx4[k], ny = y + dy4[k];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int q = ny * W + nx;
                if (elem[q] == 0 && on(q)) { elem[q] = fid; st.push_back(q); }
            }
        }
    }

    // 4. Relabel elements as 1..count in scan order.
    Pixels ret(visible.wh);
    unordered_map<uint32_t, uint32_t> label;
    element_of_label.assign(1, 0);
    count = 0;
    for (int p = 0; p < N; p++) {
        if (elem[p] == 0) continue;
        auto it = label.find(elem[p]);
        uint32_t l;
        if (it == label.end()) {
            l = ++count;
            label[elem[p]] = l;
            element_of_label.push_back(elem[p] < FALLBACK_BASE ? elem[p] : 0);
        }
        else l = it->second;
        ret.set_pixel_carelessly(p % W, p / W, l);
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

