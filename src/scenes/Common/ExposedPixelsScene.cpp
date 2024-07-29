#pragma once

#include "../Scene.cpp"

// The idea here is that the user does whatever they want with the Pixels object, manually
class ExposedPixelsScene : public Scene {
public:
    ExposedPixelsScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {}

    bool scene_requests_rerender() const override { return false; }
    void draw() override{ }
};
