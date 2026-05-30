#include "Mp4Scene.h"
#include <iostream>

using namespace std;

extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);

Mp4Scene::Mp4Scene(
    const vector<string>& mp4_filenames,
    const double playback_speed,
    const Mp4EndBehavior behavior,
    const vec2& dimensions)
:
    Scene(dimensions),
    first_frame_this_video(0),
    current_video_index(0),
    video_filenames(mp4_filenames),
    current_video_reader(mp4_filenames[0]),
    end_behavior(behavior)
{
    manager.begin_timer("MP4_Frame");
    manager.set("current_frame", "<MP4_Frame> " + to_string(playback_speed * get_video_framerate_fps()) + " * .5 + floor");
}

void Mp4Scene::draw() {
    int current_frame = state["current_frame"];
    int current_frame_adjusted = current_frame - first_frame_this_video;
    cout << "rendering concatenated mp4 videos, frame " << to_string(current_frame_adjusted) << endl;

    // Load the current video frame into a Pixels object.
    Pixels frame;
    bool no_more_frames = current_video_reader.get_frame(current_frame_adjusted, get_width(), get_height(), frame);
    if (no_more_frames) {
        if (end_behavior == Mp4EndBehavior::Stop) return;
        cout << "No more frames!" << endl;
        first_frame_this_video = current_frame;
        current_video_index = (current_video_index + 1) % video_filenames.size();
        current_video_reader.change_video(video_filenames[current_video_index]);
        current_video_reader.get_frame(0, get_width(), get_height(), frame);
    }

    // Calculate the offsets to center the frame in the output
    const vec2 offset = (get_width_height() - frame.wh) / 2;

    uint32_t* frame_ptr = cuda_alloc_pixels_on_device(frame.wh.x * frame.wh.y);
    cuda_copy_pixels_to_device(frame.pixels.data(), frame.wh.x * frame.wh.y, frame_ptr);

    // Overwrite the image onto the scene's pixel buffer
    cuda_overlay(gpu_pix->get_ptr(), get_width_height(), frame_ptr, frame.wh, offset, 1.0f, 0.0f);

    cuda_free_pixels_on_device(frame_ptr);
}

const StateQuery Mp4Scene::populate_state_query() const {
    return StateQuery{"current_frame"};
}
