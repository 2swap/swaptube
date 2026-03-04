#pragma once

#include "../../IO/VideoDecoding.h"
#include "../Scene.h"
#include "../../IO/Writer.h"
#include <vector>
#include <string>

enum Mp4EndBehavior {
    Loop,
    Stop,
};

class Mp4Scene : public Scene {
public:
    Mp4Scene(const std::vector<std::string>& mp4_filenames, const double playback_speed = 1, const Mp4EndBehavior behavior = Loop, const vec2& dimensions = vec2(1, 1));

    bool check_if_data_changed() const override;
    void mark_data_unchanged() override;
    void change_data() override;
    void draw() override;
    const StateQuery populate_state_query() const override;

private:
    int first_frame_this_video = 0;
    int current_video_index = 0;
    std::vector<std::string> video_filenames;
    MP4FrameReader current_video_reader;
    Mp4EndBehavior end_behavior;
};
