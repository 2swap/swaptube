#pragma once

#include <unordered_map>
#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

class LoopAnimationScene : public Scene {
public:
    LoopAnimationScene(const vector<string>& pn, const double width = 1, const double height = 1) : Scene(width, height), picture_names(pn) {
        manager.set({
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

        int w = get_width();
        int h = get_height();

        // Retrieve or create the cached PNG image Pixels object
        if(memo_w != w || memo_h != h){
            pixel_cache.clear();
            memo_w = w;
            memo_h = h;
        }
        if(pixel_cache.find(picture_name) == pixel_cache.end()) {
            cout << "rendering png: " << picture_name << endl;
            pixel_cache[picture_name] = png_to_pix_bounding_box(picture_name, w, h);
        }
        Pixels image = pixel_cache[picture_name];

        // Calculate the position to center the image within the bounding box
        int x_offset = (get_width() - image.w) / 2;
        int y_offset = (get_height() - image.h) / 2;

        // Overwrite the scaled image onto the scene's pixel buffer
        pix.overwrite(image, x_offset, y_offset);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"animated_frame", "loop_start", "loop_length"};
    }

private:
    const vector<string> picture_names;
    int memo_w = 0;
    int memo_h = 0;
    unordered_map<string, Pixels> pixel_cache;
};

