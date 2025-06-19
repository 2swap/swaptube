#pragma once

#include "../../io/video_decoding.cpp"
#include "../Scene.cpp"

class Mp4Scene : public Scene {
public:
    // Constructor for a single mp4 filename.
    Mp4Scene(const string& mp4_filename, const double width = 1, const double height = 1)
        : Scene(width, height), current_frame_index(0), current_video_index(0) {
        video_filenames.push_back(mp4_filename);
    }

    // Constructor for multiple mp4 filenames.
    Mp4Scene(const vector<string>& mp4_filenames, const double width = 1, const double height = 1)
        : Scene(width, height), current_frame_index(0), current_video_index(0), video_filenames(mp4_filenames) { }

    // Since we want to update to a new video frame each time, we force a redraw.
    bool check_if_data_changed() const override { return true; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    // Each call to draw loads the next frame from the concatenated videos,
    // scales it to fit within the scene's bounding box,
    // and centers it just as in PngScene.
    void draw() override {
        cout << "rendering concatenated mp4 videos, frame " << to_string(current_frame_index) << endl;

        // Load the current video frame into a Pixels object.
        bool no_more_frames = false;
        Pixels frame = mp4_to_pix_bounding_box(video_filenames[current_video_index], get_width(), get_height(), current_frame_index, no_more_frames);
        if (no_more_frames) {
            frame = mp4_to_pix_bounding_box(video_filenames[current_video_index], get_width(), get_height(), current_frame_index-1, no_more_frames);
            cout << "No more frames!" << endl;
            current_frame_index = 0;
            current_video_index = (current_video_index + 1) % video_filenames.size();
        }

        // Calculate the offsets to center the frame in the output
        int x_offset = (get_width() - frame.w) / 2;
        int y_offset = (get_height() - frame.h) / 2;

        pix.overwrite(frame, x_offset, y_offset);
        current_frame_index++;
    }

    // Mp4Scene doesn't publish any additional state.
    const StateQuery populate_state_query() const override { return StateQuery{}; }

private:
    int current_frame_index;
    int current_video_index;
    vector<string> video_filenames;
};
