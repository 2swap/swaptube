#include "WhitePaperScene.h"

extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);

WhitePaperScene::WhitePaperScene(const string& prefix, const string& author, const vector<int>& page_numbers, const vec2& dimensions)
    : Scene(dimensions), prefix(prefix), author(author), page_numbers(page_numbers) {
    manager.set({
        {"completion", "0"},
        {"which_page", "1"},
        {"page_focus", "0"},
        {"crop_top", "0"},
        {"crop_bottom", "1"},
        {"crop_left", "0"},
        {"crop_right", "1"},
    });
}

void WhitePaperScene::draw() {
    // Expect files of the form prefix-0i.png
    double page_height = get_height() * .68;
    double page_width = get_width() * .8;
    int num_pages = page_numbers.size();

    double completion = state["completion"];
    int which_page = state["which_page"];
    double page_focus = state["page_focus"];

    for(int i = num_pages - 1; i >= 0; --i) {
        int page_number = page_numbers[i];
        Pixels picture;
        pdf_page_to_pix(picture, prefix, page_number);
        Pixels cropped;
        picture.crop_by_fractions(
            vec2(state["crop_top"], state["crop_left"]),
            vec2(state["crop_bottom"], state["crop_right"]),
            cropped
        );

        Pixels scaled;
        cropped.scale_to_bounding_box(page_width, page_height, scaled);

        float this_page_is_focused = (which_page == page_number ? 1.0f : 0.0f) * page_focus;
        float this_page_not_focused = 1.0f - this_page_is_focused;

        float darken_factor = 1.0f - (i * .1f) * this_page_not_focused;

        if(i != 0) scaled.darken(darken_factor);

        double this_c = clamp(completion * (num_pages - 1) - i / 2., 0, 1);
        float pages_centered = i - (num_pages - 1) / 2.0f;

        double page_focus_multiplier = cos(page_focus * 3.1415 / 2);
        if(which_page != page_number) {
            page_focus_multiplier = 1 / page_focus_multiplier;
        }

        pages_centered *= page_focus_multiplier;

        const vec2 center(.5 + pages_centered * (.08 + .08*(1-square(1-completion))),
                           (.25/sin(this_c*3.1415/2) + .3 + pages_centered*.05));
        const vec2 offset = get_width_height() * center - scaled.wh * .5;

        float angle = pages_centered * .1f * this_page_not_focused; // .1f radians per page

        uint32_t* scaled_ptr = cuda_alloc_pixels_on_device(scaled.wh.x * scaled.wh.y);
        cuda_copy_pixels_to_device(scaled.pixels.data(), scaled.wh.x * scaled.wh.y, scaled_ptr);

        // Overwrite the scaled image onto the scene's pixel buffer
        cuda_overlay(gpu_pix->get_ptr(), get_width_height(), scaled_ptr, scaled.wh, offset, 1.0f, 0.0f);

        cuda_free_pixels_on_device(scaled_ptr);
    }

    ScalingParams sp = ScalingParams(get_width_height() * vec2(1, .13));
    Pixels text_pixels = latex_to_pix("\\text{" + author + "}", sp);
    float offset_y = get_height() * smoothlerp(-1/6., .05, state["completion"]);
    uint32_t* text_ptr = cuda_alloc_pixels_on_device(text_pixels.wh.x * text_pixels.wh.y);
    cuda_copy_pixels_to_device(text_pixels.pixels.data(), text_pixels.wh.x * text_pixels.wh.y, text_ptr);
    const vec2 text_offset((get_width() - text_pixels.wh.x) / 2, offset_y);
    cuda_overlay(gpu_pix->get_ptr(), get_width_height(), text_ptr, text_pixels.wh, text_offset, 1.0f, 0.0f);
    cuda_free_pixels_on_device(text_ptr);
}

const StateQuery WhitePaperScene::populate_state_query() const {
    return StateQuery{"completion", "which_page", "page_focus",
                      "crop_top", "crop_bottom", "crop_left", "crop_right"};
}
