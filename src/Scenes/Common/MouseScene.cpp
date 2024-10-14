#pragma once

#include "scene.cpp"

class MouseScene : public Scene {
public:
    MouseScene(const double width = 1, const double height = 1) : Scene(width, height) {}

    const StateQuery populate_state_query() const override {
        return StateQuery{"mouse_x", "mouse_y"};
    }
    void mark_data_unchanged() override { }
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }
    void draw() override{
        pix.fill_rect(state["mouse_x"], state["mouse_y"], 6, 6, 0xffff0000);
    }
};
