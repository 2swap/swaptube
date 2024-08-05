#pragma once

#include "../Scene.cpp"

// The idea here is that the user does whatever they want with the Pixels object, manually
class ExposedPixelsScene : public Scene {
public:
    ExposedPixelsScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {}

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }
    bool update_data_objects_check_if_changed() override { return false; }
    Pixels& get_pixels(){return pix;}
    void draw() override{ }
};
