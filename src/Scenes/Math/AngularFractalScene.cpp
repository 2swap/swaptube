#include "AngularFractalScene.h"
#include <vector>
#include <list>
#include <string>
#include <utility>
#include <cmath>
#include "../../Host_Device_Shared/vec.h"

using std::to_string;
using std::list;
using std::pair;

AngularFractalScene::AngularFractalScene(int sz, const float width, const float height) : CoordinateScene(width, height), size(sz) {
    for(int i = 0; i < sz; i++) manager.set("angle_"+to_string(i), "0");
}

void AngularFractalScene::draw() {
    CoordinateScene::draw();
    draw_angular_fractal();
}

void AngularFractalScene::draw_angular_fractal() {
    list<pair<vec2, int>> trail;
    vec2 current_point = vec2(0,0);
    double current_angle = 0;
    trail.push_back({current_point, OPAQUE_WHITE});
    float segment_length = 1. / size;
    for(int i = 0; i < size; i++) {
        double angle = state["angle_"+to_string(i)];
        current_angle += angle;
        vec2 direction = vec2(cos(current_angle), sin(current_angle)) * (float)segment_length;
        current_point += direction;
        trail.push_back({current_point, OPAQUE_WHITE});
    }

    draw_trail(trail, 1.0);

}

const StateQuery AngularFractalScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    for (int i = 0; i < size; i++) sq.insert("angle_" + to_string(i));
    return sq;
}
