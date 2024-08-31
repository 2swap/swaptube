#pragma once

#include "scene.cpp"

class MouseScene : public Scene {
public:
    MouseScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {}

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
