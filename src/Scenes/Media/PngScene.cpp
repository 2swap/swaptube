#include "PngScene.h"
#include "../../IO/PNG.h"
#include <iostream>

using std::cout;
using std::endl;

extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);

PngScene::PngScene(string pn, const vec2& dimensions) : Scene(dimensions), picture_name(pn) {
    manager.set({
        {"crop_top", "0"},
        {"crop_bottom", "1"},
        {"crop_left", "0"},
        {"crop_right", "1"},
    });
}

void PngScene::draw() {
    cout << "rendering png: " << picture_name << endl;
    
    // Load the PNG image into a Pixels object
    Pixels image;
    png_to_pix(image, picture_name);

    Pixels cropped;
    image.crop_by_fractions(
        vec2(state["crop_top"], state["crop_left"]),
        vec2(state["crop_bottom"], state["crop_right"]),
        cropped
    );

    Pixels scaled;
    cropped.scale_to_bounding_box(get_width(), get_height(), scaled);

    // Calculate the position to center the image within the bounding box
    const vec2 offset = (get_width_height() - scaled.wh) * 0.5f;

    uint32_t* scaled_ptr = cuda_alloc_pixels_on_device(scaled.wh.x * scaled.wh.y);
    cuda_copy_pixels_to_device(scaled.pixels.data(), scaled.wh.x * scaled.wh.y, scaled_ptr);

    // Overwrite the scaled image onto the scene's pixel buffer
    cuda_overlay(gpu_pix->get_ptr(), get_width_height(), scaled_ptr, scaled.wh, offset, 1.0f, 0.0f);

    cuda_free_pixels_on_device(scaled_ptr);
}

const StateQuery PngScene::populate_state_query() const {
    return StateQuery{"crop_top", "crop_bottom", "crop_left", "crop_right"};
}
