#include "PauseScene.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <string>
#include "../../IO/VisualMedia.h"

PauseScene::PauseScene(const vec2& dimensions)
: Scene(dimensions) {
    manager.set({{"timer", "{microblock_fraction}"}});
}

void PauseScene::draw() {
    double timer_value = state["timer"]; // expected to be between 0 and 1

    const vec2 dimensions = get_dimensions();
    const vec2 center = dimensions / 2;

    int color = OPAQUE_WHITE;

    // Draw expanding borders from center of each side outward towards corners
    // At 0 timer_value nothing drawn, at 1 timer_value the fill_rect reaches the corners

    const double border_thickness = get_geom_mean_size() / 200.;

    // Top border (from center of top edge to the corners)
    const double top_width = dimensions.x * timer_value;
    const vec2 horizontal_bar_size(top_width, border_thickness);
    pix.fill_rect(vec2(center.x - top_width/2.,                  0), horizontal_bar_size, color);
    pix.fill_rect(vec2(center.x - top_width/2., dimensions.y-border_thickness), horizontal_bar_size, color);

    // Left border (from center left edge toward top-left and bottom-left corners)
    const double left_height = dimensions.y * timer_value;
    const vec2 vertical_bar_size(border_thickness, left_height);
    pix.fill_rect(vec2(                 0, center.y - left_height/2.), vertical_bar_size, color);
    pix.fill_rect(vec2(dimensions.x-border_thickness, center.y - left_height/2.), vertical_bar_size, color);
}

const StateQuery PauseScene::populate_state_query() const {
    return StateQuery{"timer"};
}
void PauseScene::mark_data_unchanged() { }
void PauseScene::change_data() { }
bool PauseScene::check_if_data_changed() const { return false; }
