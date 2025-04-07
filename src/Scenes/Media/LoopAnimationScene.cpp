#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class LoopAnimationScene : public Scene {
public:
    LoopAnimationScene(const vector<string>& pn, const double width = 1, const double height = 1) : Scene(width, height), picture_names(pn) {
        state_manager.set({
            {"animated_frame", "<frame_number> 2 / floor"},
            {"loop_start", "0"},
            {"loop_length", to_string(pn.size())},
        });
    }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    void draw() override {
        const int frame_no = state["loop_start"] + extended_mod(state["animated_frame"], state["loop_length"]);
        if(frame_no < 0 || frame_no >= picture_names.size())
            throw runtime_error("Looped animation frame out of bounds!");
        const string& picture_name = picture_names[frame_no];

        cout << "rendering png: " << picture_name << endl;
        
        // Load the PNG image into a Pixels object
        Pixels image = png_to_pix_bounding_box(picture_name, get_width(), get_height());

        // Calculate the position to center the image within the bounding box
        int x_offset = (get_width() - image.w) / 2;
        int y_offset = (get_height() - image.h) / 2;

        // Overwrite the scaled image onto the scene's pixel buffer
        pix.overwrite(image, x_offset, y_offset);
    }

    void on_end_transition(bool is_macroblock) override { }
    const StateQuery populate_state_query() const override {
        return StateQuery{"animated_frame", "loop_start", "loop_length"};
    }

private:
    const vector<string> picture_names;
};

