#pragma once

#include "../Scene.cpp"

// The idea here is that the user does whatever they want with the Pixels object, manually
class ExposedPixelsScene : public Scene {
public:
    Pixels exposed_pixels;
    ExposedPixelsScene(const double width = 1, const double height = 1) : Scene(width, height) {
        exposed_pixels = Pixels(width * VIDEO_WIDTH, height * VIDEO_HEIGHT);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }
    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return true; }
    void draw() override{pix = exposed_pixels;}
};
