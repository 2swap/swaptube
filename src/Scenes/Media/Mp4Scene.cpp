#pragma once

#include "../../io/video_decoding.cpp"
#include "../Scene.cpp"

class Mp4Scene : public Scene {
public:
    // The constructor takes the mp4 filename along with optional width and height.
    Mp4Scene(const string& mp4_filename, const double width = 1, const double height = 1)
        : Scene(width, height), video_filename(mp4_filename), current_frame_index(0) {
        // Draw the first frame immediately
        draw();
    }

    // Since we want to update to a new video frame each time, we force a redraw.
    bool check_if_data_changed() const override { return true; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    // Each call to draw loads the next frame of the mp4 video,
    // scales it to fit within the scene's bounding box,
    // and centers it just as in PngScene.
    void draw() override {
        GUI.log_window.log("rendering mp4: " + video_filename + ", frame " + to_string(current_frame_index));
        
        // Load the current video frame into a Pixels object.
        // Assumes a function mp4_to_pix_bounding_box that takes the filename, target width,
        // target height, and the frame index.
        Pixels frame = mp4_to_pix_bounding_box(video_filename, get_width(), get_height(), current_frame_index);
        
        // Calculate the offsets to center the frame in the output
        int x_offset = (get_width() - frame.w) / 2;
        int y_offset = (get_height() - frame.h) / 2;
        
        // Overwrite the scene's pixel buffer with the video frame
        pix.overwrite(frame, x_offset, y_offset);
        
        // Increment the frame counter so that the next call draws the next frame.
        current_frame_index++;
    }

    // Mp4Scene doesn't publish any additional state.
    const StateQuery populate_state_query() const override { return StateQuery{}; }

private:
    string video_filename;
    int current_frame_index;
};

