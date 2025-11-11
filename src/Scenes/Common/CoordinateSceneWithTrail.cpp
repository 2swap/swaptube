#pragma once

#include "CoordinateScene.cpp"
#include <vector>

class CoordinateSceneWithTrail : public CoordinateScene {
public:
    int trail_color = OPAQUE_WHITE;
    list<pair<glm::vec2, int>> trail;
    CoordinateSceneWithTrail(const double width = 1, const double height = 1)
        : CoordinateScene(width, height) {
        manager.set({{"trail_opacity", "1"},
                   {"trail_x", "0"},
                   {"trail_y", "0"}});
    }

    void draw() override {
        CoordinateScene::draw();
        draw_trail(trail, state["trail_opacity"]);
        glm::vec2 vec = point_to_pixel(glm::vec2(state["trail_x"], state["trail_y"]));
        draw_point(vec, trail_color, state["trail_opacity"]);
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        sq.insert("trail_opacity");
        sq.insert("trail_x");
        sq.insert("trail_y");
        return sq;
    }

    void change_data() override {
        if(state["trail_opacity"] > 0.01)
            trail.push_back(make_pair(glm::vec2(state["trail_x"], state["trail_y"]), trail_color));
        else trail.clear();
    }

    void clear_trail() {
        trail.clear();
    }

    bool check_if_data_changed() const override { return true; }
};

