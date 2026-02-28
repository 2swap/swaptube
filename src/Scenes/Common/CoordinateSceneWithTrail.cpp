#include "CoordinateSceneWithTrail.h"
#include <vector>

CoordinateSceneWithTrail::CoordinateSceneWithTrail(const double width, const double height)
    : CoordinateScene(width, height) {
    manager.set({{"trail_opacity", "1"},
               {"trail_x", "0"},
               {"trail_y", "0"}});
}

void CoordinateSceneWithTrail::draw() {
    CoordinateScene::draw();
    draw_trail(trail, state["trail_opacity"]);
    vec2 vec = point_to_pixel(vec2(state["trail_x"], state["trail_y"]));
    draw_point(vec, trail_color, state["trail_opacity"]);
}

const StateQuery CoordinateSceneWithTrail::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    sq.insert("trail_opacity");
    sq.insert("trail_x");
    sq.insert("trail_y");
    return sq;
}

void CoordinateSceneWithTrail::change_data() {
    if(state["trail_opacity"] > 0.01)
        trail.push_back(make_pair(vec2(state["trail_x"], state["trail_y"]), trail_color));
    else trail.clear();
}

void CoordinateSceneWithTrail::clear_trail() {
    trail.clear();
}

bool CoordinateSceneWithTrail::check_if_data_changed() const { return true; }
