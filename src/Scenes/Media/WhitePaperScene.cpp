#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

// HOW TO MAKE PAGES:
// pdftocairo -png -f 1 -l 3 -r 300 paper.pdf prefix
// (This makes 3 pages at 300 DPI, named prefix-01.png, prefix-02.png, prefix-03.png)

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

class WhitePaperScene : public Scene {
public:
    WhitePaperScene(const string& prefix, const int np, const double width = 1, const double height = 1)
        : Scene(width, height), prefix(prefix), num_pages(np) {
        manager.set("completion", "0");
    }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        // Expect 3 files of the form prefix-0i.png
        double page_height = get_height() * .8;
        double page_width = 5000;
        for(int i = num_pages; i >= 1; --i) {
            string picture_name = prefix + "-" + (i < 10 ? "0" : "") + to_string(i);
            Pixels picture = png_to_pix_bounding_box(picture_name, page_width, page_height);
            if(i != 1) picture.darken(1.0f - (i-1) * .1f);
            // Center the scaled image within the scene
            double c = state["completion"];
            double this_c = clamp(c * (num_pages - 1) - (i - 1) / 2., 0, 1);
            float pages_centered = i - (num_pages + 1) / 2.0f;
            float x_center = (.5 + pages_centered * (.08 + .08*(1-square(1-c))));
            float y_center = (.25/sin(this_c*3.1415/2)+.25 + pages_centered*.05);
            float x_offset = get_width() * x_center - picture.w * .5;
            float y_offset = get_height() * y_center - picture.h * .5;

            cuda_overlay(pix.pixels.data(), pix.w, pix.h,
                         picture.pixels.data(), picture.w, picture.h,
                         (int)x_offset, (int)y_offset,
                         1.0f);
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"completion"};
    }

private:
    const string prefix;
    const int num_pages;
};
