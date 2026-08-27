#include "PngScene.h"
#include "../../IO/PNG.h"
#include <iostream>

using std::cout;
using std::endl;

extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);
extern "C" void cuda_crop_scale_darken_device(
    const uint32_t* d_input, const ivec2& in_wh,
    uint32_t* d_output, const ivec2& out_wh,
    const vec2& crop_tl, const vec2& crop_br,
    const float darken_factor);

PngScene::PngScene(string pn, const vec2& dimensions) : Scene(dimensions), picture_name(pn) {
    manager.set({
        {"crop_top", "0"},
        {"crop_bottom", "1"},
        {"crop_left", "0"},
        {"crop_right", "1"},
    });

    cout << "rendering png: " << picture_name << endl;
    Pixels image;
    png_to_pix(image, picture_name);
    cached_image = make_unique<DevicePointer>(image.wh);
    cached_image->copy_to_device(image.pixels.data());
}

void PngScene::draw() {
    const ivec2 src_wh = cached_image->get_wh();
    const vec2 crop_tl(state["crop_top"], state["crop_left"]);
    const vec2 crop_br(state["crop_bottom"], state["crop_right"]);

    const ivec2 cropped_wh(
        max(1, (int)floor(src_wh.x * crop_br.x) - (int)floor(src_wh.x * crop_tl.x)),
        max(1, (int)floor(src_wh.y * crop_br.y) - (int)floor(src_wh.y * crop_tl.y))
    );
    const float scale = min((float)get_width() / cropped_wh.x, (float)get_height() / cropped_wh.y);
    const ivec2 dest_wh(max(1, (int)(cropped_wh.x * scale)), max(1, (int)(cropped_wh.y * scale)));

    DevicePointer scaled(dest_wh);
    cuda_crop_scale_darken_device(cached_image->get_ptr(), src_wh, scaled.get_ptr(), dest_wh, crop_tl, crop_br, 1.0f);

    // Overwrite the scaled image onto the scene's pixel buffer
    const vec2 offset = get_width_height() * 0.5f;
    cuda_overlay(gpu_pix.get_ptr(), get_width_height(), scaled.get_ptr(), dest_wh, offset, 1.0f, 0.0f);
}
