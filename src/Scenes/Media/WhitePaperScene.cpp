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
    WhitePaperScene(const string& prefix, const double width = 1, const double height = 1)
        : Scene(width, height), prefix(prefix) {
        manager.set("completion", "0");
    }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        // Expect a number of files of the form {prefix}-01.png, {prefix}-02.png, ...
        double page_height = get_height() * .8;
        double page_width = 5000;
        for(int i = 3; i >= 1; --i) {
            string picture_name = prefix + "-" + (i < 10 ? "0" : "") + to_string(i);
            Pixels picture = png_to_pix_bounding_box(picture_name, page_width, page_height);
            if(i != 1) picture.darken(1.0f - (i-1) * .1f);
            // Center the scaled image within the scene
            double c = state["completion"];
            double this_c = clamp(c * 3 - (i - 1), 0, 1);
            float x_offset = get_width()  * (.5 + (i-2) * .1)            - picture.w * .5;
            float y_offset = get_height() * (.3 + .2/this_c + (i-2)*.05) - picture.h * .5;

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
    string prefix;
};
