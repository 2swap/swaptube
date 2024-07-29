#pragma once

#include "scene.cpp"

class MouseScene : public Scene {
public:
    MouseScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {
        append_to_state_query(StateQuery{"mouse_x", "mouse_y"})
    }

    bool scene_requests_rerender() const override { return false; }
    void draw() override{
        pix.fill(TRANSPARENT_BLACK);
        pix.fill_rect(state["mouse_x"], state["mouse_y"], 6, 6, 0xffff0000);
    }
};
