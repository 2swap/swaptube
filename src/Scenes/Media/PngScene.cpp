#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

class PngScene : public Scene {
public:
    PngScene(string pn, const double width = 1, const double height = 1) : Scene(width, height), picture_name(pn) {
        manager.set({
            {"crop_top", "0"},
            {"crop_bottom", "0"},
            {"crop_left", "0"},
            {"crop_right", "0"},
        });
    }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        cout << "rendering png: " << picture_name << endl;
        
        // Load the PNG image into a Pixels object
        Pixels image;
        png_to_pix(image, picture_name);

        Pixels cropped;
        image.crop_by_fractions(
            state["crop_top"],
            state["crop_bottom"],
            state["crop_left"],
            state["crop_right"],
            cropped
        );

        Pixels scaled;
        cropped.scale_to_bounding_box(get_width(), get_height(), scaled);

        // Calculate the position to center the image within the bounding box
        int x_offset = (get_width() - scaled.w) / 2;
        int y_offset = (get_height() - scaled.h) / 2;

        // Overwrite the scaled image onto the scene's pixel buffer
        pix.overwrite(scaled, x_offset, y_offset);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"crop_top", "crop_bottom", "crop_left", "crop_right"};
    }

private:
    string picture_name;
};

