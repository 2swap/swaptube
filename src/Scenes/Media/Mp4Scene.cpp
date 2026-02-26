#include "Mp4Scene.h"
#include <iostream>

using namespace std;

Mp4Scene::Mp4Scene(const vector<string>& mp4_filenames, const double playback_speed, const double width, const double height)
    : Scene(width, height), first_frame_this_video(0), current_video_index(0), video_filenames(mp4_filenames), current_video_reader(mp4_filenames[0]) {
    manager.begin_timer("MP4_Frame");
    manager.set("current_frame", "<MP4_Frame> " + to_string(playback_speed * get_video_framerate_fps()) + " * .5 + floor");
}

bool Mp4Scene::check_if_data_changed() const { return false; }
void Mp4Scene::mark_data_unchanged() { }
void Mp4Scene::change_data() { }

void Mp4Scene::draw() {
    int current_frame = state["current_frame"];
    int current_frame_adjusted = current_frame - first_frame_this_video;
    cout << "rendering concatenated mp4 videos, frame " << to_string(current_frame_adjusted) << endl;

    // Load the current video frame into a Pixels object.
    Pixels frame;
    bool no_more_frames = current_video_reader.get_frame(current_frame_adjusted, get_width(), get_height(), frame);
    if (no_more_frames) {
        cout << "No more frames!" << endl;
        first_frame_this_video = current_frame;
        current_video_index = (current_video_index + 1) % video_filenames.size();
        current_video_reader.change_video(video_filenames[current_video_index]);
        current_video_reader.get_frame(0, get_width(), get_height(), frame);
    }

    // Calculate the offsets to center the frame in the output
    int x_offset = (get_width() - frame.w) / 2;
    int y_offset = (get_height() - frame.h) / 2;

    pix.overwrite(frame, x_offset, y_offset);
}

const StateQuery Mp4Scene::populate_state_query() const {
    return StateQuery{"current_frame"};
}
