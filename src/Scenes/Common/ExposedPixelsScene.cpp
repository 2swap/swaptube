#include "ExposedPixelsScene.h"

ExposedPixelsScene::ExposedPixelsScene(const vec2& dimensions) : Scene(dimensions) {
    exposed_pixels = Pixels(get_width(), get_height());
}

const StateQuery ExposedPixelsScene::populate_state_query() const {
    return StateQuery{};
}
void ExposedPixelsScene::draw() { pix = exposed_pixels; }
