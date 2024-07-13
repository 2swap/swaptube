#pragma once

#include "scene.cpp"

// The idea here is that the user does whatever they want with the Pixels object, manually
class ExposedPixelsScene : public Scene {
public:
    ExposedPixelsScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    void query(bool& done_scene, Pixels*& p) override {
        p = &pix;
    }
};
