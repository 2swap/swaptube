#pragma once

#include "scene.cpp"

class MouseScene : public Scene {
public:
    MouseScene(const int width, const int height) : Scene(width, height) {}
    MouseScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    void query(bool& done_scene, Pixels*& p) override {
        pix.fill(TRANSPARENT_BLACK);
        pix.fill_rect(dag["mouse_x"], dag["mouse_y"], 2, 2, 0xffff0000);
        p = &pix;
        done_scene = scene_duration_frames <= time;
        time++;
    }
};
