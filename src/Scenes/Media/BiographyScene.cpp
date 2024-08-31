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

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void set_bio_text(const vector<string>& new_text) {
        bio_text = new_text;
        draw(); // Redraw the scene with the updated text
    }

    void draw() override {
        cout << "Rendering BiographyScene: " << picture_name << endl;

        // Render the person's picture as a PNG
        Pixels picture = png_to_pix(picture_name);

        // Calculate the scaling factor to fit the image within the top portion of the scene
        int pic_height = h * 2 / 3;
        float scale = min(static_cast<float>(w) / picture.w, static_cast<float>(pic_height) / picture.h);

        int new_width = static_cast<int>(picture.w * scale);
        int new_height = static_cast<int>(picture.h * scale);

        Pixels scaled_picture = picture.bicubic_scale(new_width, new_height);

        // Center the scaled image within the scene
        int x_offset = (w - new_width) / 2;
        int y_offset = (h - pic_height) / 2;

        pix.fill(TRANSPARENT_BLACK); // Clear the scene
        pix.overwrite(scaled_picture, x_offset, y_offset); // Draw the picture

        // Render the biography text below the picture
        int text_y = y_offset + new_height + 20; // Position the text below the image, with a small margin
        int line_height = 30; // Approximate height for each line of text

        for (const string& line : bio_text) {
            // Render the line of text (assuming a simple method to convert text to a Pixels object exists)
            ScalingParams sp = ScalingParams(w, line_height);
            Pixels text_pixels = eqn_to_pix(line, sp);
            pix.overwrite(text_pixels, (w - text_pixels.w) / 2, text_y);

            text_y += line_height;
        }
    }

    void on_end_transition() override { }
    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

private:
    string picture_name;
    vector<string> bio_text;
};
