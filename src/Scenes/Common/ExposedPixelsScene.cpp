#include "ExposedPixelsScene.h"

ExposedPixelsScene::ExposedPixelsScene(const vec2& dimensions) : Scene(dimensions) {
    exposed_pixels = Pixels(get_dimensions());
}

const StateQuery ExposedPixelsScene::populate_state_query() const {
    return StateQuery{};
}
void ExposedPixelsScene::mark_data_unchanged() { }
void ExposedPixelsScene::change_data() { }
bool ExposedPixelsScene::check_if_data_changed() const { return true; }
void ExposedPixelsScene::draw() { pix = exposed_pixels; }
