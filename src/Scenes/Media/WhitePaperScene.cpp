#include "WhitePaperScene.h"

extern "C" void cuda_overlay_with_rotation (
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity, const float angle);

WhitePaperScene::WhitePaperScene(const string& prefix, const string& author, const vector<int>& page_numbers, const vec2& dimensions)
    : Scene(dimensions), prefix(prefix), author(author), page_numbers(page_numbers) {
    manager.set({
        {"completion", "0"},
        {"which_page", "1"},
        {"page_focus", "0"},
        {"crop_top", "0"},
        {"crop_bottom", "0"},
        {"crop_left", "0"},
        {"crop_right", "0"},
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
            state["crop_top"],
            state["crop_bottom"],
            state["crop_left"],
            state["crop_right"],
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

        float x_center = (.5 + pages_centered * (.08 + .08*(1-square(1-completion))));
        float y_center = (.25/sin(this_c*3.1415/2) + .3 + pages_centered*.05);
        float x_offset = get_width() * x_center - scaled.w * .5;
        float y_offset = get_height() * y_center - scaled.h * .5;

        float angle = pages_centered * .1f * this_page_not_focused; // .1f radians per page

        cuda_overlay_with_rotation(pix.pixels.data(), pix.w, pix.h,
                     scaled.pixels.data(), scaled.w, scaled.h,
                     (int)x_offset, (int)y_offset,
                     1.0f, angle
        );
    }

    ScalingParams sp = ScalingParams(get_width(), get_height()/6);
    Pixels text_pixels = latex_to_pix("\\text{" + author + "}", sp);
    int offset_y = get_height() * smoothlerp(-1/6., .05, state["completion"]);
    cuda_overlay(pix.pixels.data(), pix.w, pix.h,
                 text_pixels.pixels.data(), text_pixels.w, text_pixels.h,
                 (int)((get_width() - text_pixels.w) / 2), offset_y,
                 1.0f
    );
}

const StateQuery WhitePaperScene::populate_state_query() const {
    return StateQuery{"completion", "which_page", "page_focus",
                      "crop_top", "crop_bottom", "crop_left", "crop_right"};
}
