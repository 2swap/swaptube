#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

class WhitePaperScene : public Scene {
public:
    WhitePaperScene(const string& prefix, const vector<int>& page_numbers, const double width = 1, const double height = 1)
        : Scene(width, height), prefix(prefix), page_numbers(page_numbers) {
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

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        // Expect files of the form prefix-0i.png
        double page_height = get_height() * .8;
        double page_width = get_width() * .8;
        int num_pages = page_numbers.size();
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

            if(i != 1) scaled.darken(1.0f - i * .1f);

            // Center the scaled image within the scene
            double completion = state["completion"];
            int which_page = state["which_page"];
            double page_focus = state["page_focus"];

            double this_c = clamp(completion * (num_pages - 1) - i / 2., 0, 1);
            float pages_centered = i + 1 - (num_pages + 1) / 2.0f;

            double page_focus_multiplier = cos(page_focus * 3.1415 / 2);
            if(which_page != page_number) {
                page_focus_multiplier = 1 / page_focus_multiplier;
            }

            pages_centered *= page_focus_multiplier;

            float x_center = (.5 + pages_centered * (.08 + .08*(1-square(1-completion))));
            float y_center = (.25/sin(this_c*3.1415/2) + .25 + pages_centered*.05);
            float x_offset = get_width() * x_center - scaled.w * .5;
            float y_offset = get_height() * y_center - scaled.h * .5;

            cuda_overlay(pix.pixels.data(), pix.w, pix.h,
                         scaled.pixels.data(), scaled.w, scaled.h,
                         (int)x_offset, (int)y_offset,
                         1.0f);
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"completion", "which_page", "page_focus",
                          "crop_top", "crop_bottom", "crop_left", "crop_right"};
    }

private:
    const string prefix;
    const vector<int> page_numbers;
};
