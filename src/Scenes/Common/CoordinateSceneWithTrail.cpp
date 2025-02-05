#pragma once

#include "CoordinateScene.cpp"
#include <vector>

class CoordinateSceneWithTrail : public CoordinateScene {
public:
    int trail_color = OPAQUE_WHITE;
    vector<pair<double, double>> trail;
    CoordinateSceneWithTrail(const double width = 1, const double height = 1)
        : CoordinateScene(width, height) {
        state_manager.add_equation("trail_opacity", "0");
        state_manager.add_equation("trail_x", "0");
        state_manager.add_equation("trail_y", "0");
    }

    void draw() override {
        CoordinateScene::draw();
        draw_trail(trail, trail_color, state["trail_opacity"]);
        draw_point(make_pair(state["trail_x"], state["trail_y"]), trail_color, state["trail_opacity"]);
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        sq.insert("trail_opacity");
        sq.insert("trail_x");
        sq.insert("trail_y");
        return sq;
    }

    void change_data() override { if(state["trail_opacity"] > 0.01) trail.push_back(make_pair(state["trail_x"], state["trail_y"])); else trail.clear();}
    bool check_if_data_changed() const override { return true; }
};

