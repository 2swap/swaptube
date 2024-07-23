#pragma once

#include "scene.cpp"

class MouseScene : public Scene {
public:
    MouseScene(const int width, const int height) : Scene(width, height) {}

    void query(Pixels*& p) override {
        pix.fill(TRANSPARENT_BLACK);
        pix.fill_rect(dag["mouse_x"], dag["mouse_y"], 6, 6, 0xffff0000);
        p = &pix;
    }
};
