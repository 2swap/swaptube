#pragma once

#include "../../io/video_decoding.cpp"
#include "../Scene.cpp"

class Mp4Scene : public Scene {
public:
    Mp4Scene(const vector<string>& mp4_filenames, const double playback_speed = 1, const double width = 1, const double height = 1)
        : Scene(width, height), first_frame_this_video(0), current_video_index(0), video_filenames(mp4_filenames), current_video_reader(mp4_filenames[0]) {
        manager.begin_timer("MP4_Frame");
        manager.set("current_frame_index", "<MP4_Frame> " + to_string(playback_speed * FRAMERATE) + " * .5 + floor");
    }

    // Since we want to update to a new video frame each time, we force a redraw.
    bool check_if_data_changed() const override { return true; }
    void mark_data_unchanged() override { }
    void change_data() override { }

    // Each call to draw loads the next frame from the concatenated videos,
    // scales it to fit within the scene's bounding box,
    // and centers it just as in PngScene.
    void draw() override {
        int current_frame_index = state["current_frame_index"];
        int current_frame_index_adjusted = current_frame_index - first_frame_this_video;
        cout << "rendering concatenated mp4 videos, frame " << to_string(current_frame_index_adjusted) << endl;

        // Load the current video frame into a Pixels object.
        bool no_more_frames = false;
        Pixels frame = current_video_reader.get_frame(current_frame_index_adjusted, get_width(), get_height(), no_more_frames);
        if (no_more_frames) {
            cout << "No more frames!" << endl;
            first_frame_this_video = current_frame_index;
            current_video_index = (current_video_index + 1) % video_filenames.size();
            current_video_reader.change_video(video_filenames[current_video_index]);
            frame = current_video_reader.get_frame(0, get_width(), get_height(), no_more_frames);
        }

        // Calculate the offsets to center the frame in the output
        int x_offset = (get_width() - frame.w) / 2;
        int y_offset = (get_height() - frame.h) / 2;

        pix.overwrite(frame, x_offset, y_offset);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"current_frame_index"};
    }

private:
    int first_frame_this_video = 0;
    int current_video_index = 0;
    vector<string> video_filenames;
    MP4FrameReader current_video_reader;
};
