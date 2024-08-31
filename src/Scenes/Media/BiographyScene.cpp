#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"
#include "PngScene.cpp"

class BiographyScene : public Scene {
public:
    BiographyScene(string picture, const vector<string>& biography_text, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), picture_name(picture), bio_text(biography_text) {
        draw();
    }

    bool check_if_data_changed() const override { return text_added; }
    void mark_data_unchanged() override { text_added = false; }
    void change_data() override { }

    void set_bio_text(const vector<string>& new_text) {
        bio_text = new_text;
        text_added = true;
    }

    void append_bio_text(const string& new_text) {
        bio_text.push_back(new_text);;
        text_added = true;
    }

    void draw() override {
        // Render the person's picture as a PNG
        Pixels picture = png_to_pix(picture_name);

        // Calculate the scaling factor to fit the image within the top portion of the scene
        int pic_height = get_height() * 2 / 3;
        float scale = min(static_cast<float>(get_width()) / picture.w, static_cast<float>(pic_height) / picture.h);

        float new_width  = picture.w * scale;
        float new_height = picture.h * scale;

        Pixels scaled_picture = picture.bicubic_scale(new_width, new_height);

        // Center the scaled image within the scene
        float x_offset = (get_width() - new_width) / 2.;
        float y_offset = (get_height() - new_height) / 4.;

        pix.overwrite(scaled_picture, x_offset, y_offset); // Draw the picture

        // Render the biography text below the picture
        float text_y = y_offset*1.1 + pic_height; // Position the text below the image, with a small margin
        float line_height = y_offset; // Approximate height for each line of text

        for (const string& line : bio_text) {
            ScalingParams sp = ScalingParams(get_width(), line_height);
            Pixels text_pixels = eqn_to_pix(latex_text(line), sp);
            pix.overwrite(text_pixels, (get_width() - text_pixels.w) / 2, text_y);

            text_y += line_height*1.1/2;
        }
    }

    void on_end_transition() override { }
    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

private:
    bool text_added = true;
    string picture_name;
    vector<string> bio_text;
};
