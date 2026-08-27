#include "WhitePaperScene.h"

extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);
extern "C" void cuda_crop_scale_darken_device(
    const uint32_t* d_input, const ivec2& in_wh,
    uint32_t* d_output, const ivec2& out_wh,
    const vec2& crop_tl, const vec2& crop_br,
    const float darken_factor);

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

    for (int page_number : page_numbers) {
        Pixels picture;
        pdf_page_to_pix(picture, prefix, page_number);
        auto dp = make_unique<DevicePointer>(picture.wh);
        dp->copy_to_device(picture.pixels.data());
        page_gpu.push_back(std::move(dp));
    }
}

void WhitePaperScene::draw() {
    // Expect files of the form prefix-0i.png
    double page_height = get_height() * .68;
    double page_width = get_width() * .8;
    int num_pages = page_numbers.size();

    double completion = state["completion"];
    int which_page = state["which_page"];
    double page_focus = state["page_focus"];

    const vec2 crop_tl(state["crop_left"], state["crop_top"]);
    const vec2 crop_br(state["crop_right"], state["crop_bottom"]);

    for(int i = num_pages - 1; i >= 0; --i) {
        int page_number = page_numbers[i];
        const ivec2 src_wh = page_gpu[i]->get_wh();

        const vec2 cropped_wh_raw = src_wh * (crop_br - crop_tl);
        const vec2 cropped_wh(max(1.0f, cropped_wh_raw.x), max(1.0f, cropped_wh_raw.y));
        const float scale = min(page_width / cropped_wh.x, page_height / cropped_wh.y);
        const ivec2 dest_wh(max(1, (int)(cropped_wh.x * scale)), max(1, (int)(cropped_wh.y * scale)));

        float this_page_is_focused = (which_page == page_number ? 1.0f : 0.0f) * page_focus;
        float this_page_not_focused = 1.0f - this_page_is_focused;

        float darken_factor = 1.0f - (i * .1f) * this_page_not_focused;
        if(i == 0) darken_factor = 1.0f;

        DevicePointer scaled(dest_wh);
        cuda_crop_scale_darken_device(page_gpu[i]->get_ptr(), src_wh, scaled.get_ptr(), dest_wh, crop_tl, crop_br, darken_factor);

        double this_c = clamp(completion * (num_pages - 1) - i / 2., 0, 1);
        float pages_centered = i - (num_pages - 1) / 2.0f;

        double page_focus_multiplier = cos(page_focus * 3.1415 / 2);
        if(which_page != page_number) {
            page_focus_multiplier = 1 / page_focus_multiplier;
        }

        pages_centered *= page_focus_multiplier;

        const vec2 center(.5 + pages_centered * (.08 + .08*(1-square(1-completion))),
                           (.25/sin(this_c*3.1415/2) + .3 + pages_centered*.035));

        float angle = pages_centered * .1f * this_page_not_focused; // .1f radians per page

        // Overwrite the scaled image onto the scene's pixel buffer
        const vec2 offset = get_width_height() * center;
        cuda_overlay(gpu_pix.get_ptr(), get_width_height(), scaled.get_ptr(), dest_wh, offset, 1.0f, angle);
    }

    const vec2 author_offset = get_width_height() * vec2(.5, smoothlerp(-.1, .07, state["completion"]));
    write_text(gpu_pix.get_ptr(), gpu_pix.get_wh(), "\\text{" + author + "}", author_offset, vec2(1, .13)*get_width_height(), 1.0f, 0.0f);
}
