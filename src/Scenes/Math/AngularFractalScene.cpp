#pragma once

#include "../Common/CoordinateScene.cpp"
#include <vector>

class AngularFractalScene : public CoordinateScene {
private:
    const int size;

public:
    AngularFractalScene(int sz, const float width = 1, const float height = 1) : CoordinateScene(width, height), size(sz) {
        for(int i = 0; i < sz; i++) state.set("angle_"+to_string(i), "0");
    }

    void draw() override {
        CoordinateScene::draw();
        draw_angular_fractal();
    }

    void draw_angular_fractal() {
        list<pair<glm::vec2, int>> trail;
        glm::vec2 current_point = glm::vec2(0,0);
        double current_angle = 0;
        trail.push_back({current_point, OPAQUE_WHITE});
        float segment_length = 1. / size;
        for(int i = 0; i < size; i++) {
            double angle = state["angle_"+to_string(i)];
            current_angle += angle;
            glm::vec2 direction = glm::vec2(cos(current_angle), sin(current_angle)) * (float)segment_length;
            current_point += direction;
            trail.push_back({current_point, OPAQUE_WHITE});
        }

        draw_trail(trail, 1.0);

    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        for (int i = 0; i < size; i++) sq.insert("angle_" + to_string(i));
        return sq;
    }

    //void mark_data_unchanged() override { }
    //void change_data() override { }
    //bool check_if_data_changed() const override { }
};
