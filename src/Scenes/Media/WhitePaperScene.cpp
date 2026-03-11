#include "WhitePaperScene.h"

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

bool WhitePaperScene::check_if_data_changed() const { return false; }
void WhitePaperScene::mark_data_unchanged() { }
void WhitePaperScene::change_data() { }

void WhitePaperScene::draw() {
    // Expect files of the form prefix-0i.png
    cout << "A" << endl;
    const vec2 page_size = get_dimensions() * vec2(.8, .68);
    const int num_pages = page_numbers.size();

    const double completion = state["completion"];
    const int which_page = state["which_page"];
    const double page_focus = state["page_focus"];

    for(int i = num_pages - 1; i >= 0; --i) {
        int page_number = page_numbers[i];
        Pixels picture;
        pdf_page_to_pix(picture, prefix, page_number);
        Pixels cropped;
        picture.crop_by_fractions(
            vec2(state["crop_left"], state["crop_top"]),
            vec2(state["crop_right"], state["crop_bottom"]),
            cropped
        );

        Pixels scaled;
        cropped.scale_to_bounding_box(page_size, scaled);

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
        vec2 center = vec2(x_center, y_center);
        vec2 offset = get_dimensions() * center - scaled.size * .5;

        float angle = pages_centered * .1f * this_page_not_focused; // .1f radians per page

        cuda_overlay(pix.pixels.data(), pix.size,
                     scaled.pixels.data(), scaled.size,
                     offset, 1.0f, angle
        );
    }

    ScalingParams sp = ScalingParams(get_dimensions() / vec2(1, 6));
    Pixels text_pixels = latex_to_pix("\\text{" + author + "}", sp);
    int offset_x = (get_width() - text_pixels.size.x) / 2;
    int offset_y = get_height() * smoothlerp(-1/6., .05, state["completion"]);
    cuda_overlay(pix.pixels.data(), pix.size,
                 text_pixels.pixels.data(), text_pixels.size,
                 vec2(offset_x, offset_y),
                 1.0f, 0
    );
}

const StateQuery WhitePaperScene::populate_state_query() const {
    return StateQuery{"completion", "which_page", "page_focus",
                      "crop_top", "crop_bottom", "crop_left", "crop_right"};
}
