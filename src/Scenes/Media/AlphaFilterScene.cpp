#include "AlphaFilterScene.h"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <string>

AlphaFilterScene::AlphaFilterScene(
    std::shared_ptr<Scene> sc,
    const unsigned int preserve_col,
    const vec2& dimensions) : SuperScene(dimensions), preserve_col(preserve_col)
{
    add_subscene_check_dupe("main", sc);
}

int coldist(unsigned int col1, unsigned int col2) {
    int a1 = (col1 >> 24) & 0xFF;
    int r1 = (col1 >> 16) & 0xFF;
    int g1 = (col1 >> 8) & 0xFF;
    int b1 = col1 & 0xFF;

    int a2 = (col2 >> 24) & 0xFF;
    int r2 = (col2 >> 16) & 0xFF;
    int g2 = (col2 >> 8) & 0xFF;
    int b2 = col2 & 0xFF;

    return abs(a1 - a2) + abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2);
}

void AlphaFilterScene::draw() {
    std::shared_ptr<Scene> subscene = subscenes["main"];

    Pixels* subscene_pix = nullptr;
    subscene->query(subscene_pix);

    int w = get_width();
    int h = get_height();

    for(int x = 0; x < pix.w; x++) {
        for(int y = 0; y < pix.h; y++) {
            unsigned int col = subscene_pix->get_pixel_carelessly(x, y);
            int dist = coldist(col, preserve_col);
            if (dist < 256) pix.set_pixel_carelessly(x, y, col);
        }
    }
}
