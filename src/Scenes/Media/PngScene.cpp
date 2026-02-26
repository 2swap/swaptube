#include "PngScene.h"
#include "../../IO/VisualMedia.h"
#include <iostream>

using std::cout;
using std::endl;

PngScene::PngScene(string pn, const double width, const double height) : Scene(width, height), picture_name(pn) {
    manager.set({
        {"crop_top", "0"},
        {"crop_bottom", "0"},
        {"crop_left", "0"},
        {"crop_right", "0"},
    });
}

bool PngScene::check_if_data_changed() const { return false; }
void PngScene::mark_data_unchanged() { }
void PngScene::change_data() { }

void PngScene::draw() {
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

const StateQuery PngScene::populate_state_query() const {
    return StateQuery{"crop_top", "crop_bottom", "crop_left", "crop_right"};
}
