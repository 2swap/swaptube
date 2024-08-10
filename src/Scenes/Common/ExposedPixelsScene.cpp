#pragma once

#include "../Scene.cpp"

// The idea here is that the user does whatever they want with the Pixels object, manually
class ExposedPixelsScene : public Scene {
public:
    ExposedPixelsScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {}

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }
    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return false; }
    Pixels& get_pixels(){return pix;}
    void draw() override{ }
};
