#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string pn, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height), picture_name(pn) { draw(); }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        cout << "rendering png: " << picture_name << endl;
        
        // Load the PNG image into a Pixels object
        Pixels image = png_to_pix(picture_name);

        // Calculate the scaling factor based on the bounding box
        float scale = min(static_cast<float>(w) / image.w, static_cast<float>(h) / image.h);

        // Calculate the new dimensions
        int new_width = static_cast<int>(image.w * scale);
        int new_height = static_cast<int>(image.h * scale);

        // Scale the image using bicubic interpolation
        Pixels scaled_image = image.bicubic_scale(new_width, new_height);

        // Calculate the position to center the image within the bounding box
        int x_offset = (w - new_width) / 2;
        int y_offset = (h - new_height) / 2;

        // Overwrite the scaled image onto the scene's pixel buffer
        pix.overwrite(scaled_image, x_offset, y_offset);
    }

    void on_end_transition() override { }
    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

private:
    string picture_name;
};

