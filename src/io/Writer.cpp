#pragma once

#include "AudioWriter.cpp"
#include "VideoWriter.cpp"
#include "SubtitleWriter.cpp"
#include "ShtookaWriter.cpp"

class Writer {
public:
    AVFormatContext* format_context = nullptr;
    AudioWriter* audio = nullptr;
    VideoWriter* video = nullptr;
    SubtitleWriter* subtitle = nullptr;
    ShtookaWriter* shtooka = nullptr;

    Writer() {
        const string video_path = "io_out/Video.mp4";
        int ret = avformat_alloc_output_context2(&format_context, NULL, NULL, video_path.c_str());
        if (ret < 0) throw runtime_error("Failed to allocate output format context");
        if (format_context == nullptr) throw runtime_error("Failed to allocate output format context");

        audio = new AudioWriter(format_context);
        video = new VideoWriter(format_context, video_path);
        subtitle = new SubtitleWriter();
        shtooka = new ShtookaWriter();
    }

    ~Writer() {
        delete audio;
        delete video; // This also finalizes FORMAT_CONTEXT
        delete subtitle;
        delete shtooka;
    }
};
