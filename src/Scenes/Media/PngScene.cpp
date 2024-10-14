#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string pn, const double width = 1, const double height = 1) : Scene(width, height), picture_name(pn) { draw(); }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        cout << "rendering png: " << picture_name << endl;
        
        // Load the PNG image into a Pixels object
        Pixels image = png_to_pix_bounding_box(picture_name, get_width(), get_height());

        // Calculate the position to center the image within the bounding box
        int x_offset = (get_width() - image.w) / 2;
        int y_offset = (get_height() - image.h) / 2;

        // Overwrite the scaled image onto the scene's pixel buffer
        pix.overwrite(image, x_offset, y_offset);
    }

    void on_end_transition() override { }
    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

private:
    string picture_name;
};

