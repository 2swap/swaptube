#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

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
        // Expect a number of files of the form {prefix}_01.png, {prefix}_02.png, ...
        Pixels picture = png_to_pix_bounding_box(picture_name, get_width()*.66, get_height()*.66);

        // Center the scaled image within the scene
        float x_offset = (get_width() - picture.w) / 2.;
        float y_offset = (get_height() - picture.h) / 3.;

        pix.overwrite(picture, x_offset, y_offset); // Draw the picture

        // Render the biography text below the picture
        float text_y = y_offset*1.1 + picture.h; // Position the text below the image, with a small margin
        float line_height = y_offset; // Approximate height for each line of text

        ScalingParams sp = ScalingParams(get_width(), get_height()/6);
        Pixels text_pixels = latex_to_pix("\\text{" + name + "}", sp);
        pix.overwrite(text_pixels, (get_width() - text_pixels.w) / 2, (y_offset-text_pixels.h)/2);

        for (const string& line : bio_text) {
            sp = ScalingParams(get_width(), line_height);
            text_pixels = latex_to_pix("\\text{" + line + "}", sp);
            pix.overwrite(text_pixels, (get_width() - text_pixels.w) / 2, text_y);

            text_y += line_height*1.1/2;
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

private:
    string prefix;
};
