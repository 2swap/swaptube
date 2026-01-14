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
        shtooka = new ShtookaWriter();
        subtitle = new SubtitleWriter();

        if (SMOKETEST) return;
        const string video_path = "io_out/Video.mkv";
        int ret = avformat_alloc_output_context2(&format_context, NULL, NULL, video_path.c_str());
        if (ret < 0) throw runtime_error("Failed to allocate output format context");
        if (format_context == nullptr) throw runtime_error("Failed to allocate output format context");

        audio = new AudioWriter(format_context);
        video = new VideoWriter(format_context, video_path);
    }

    ~Writer() {
        delete shtooka;
        delete subtitle;

        if (SMOKETEST) return;
        delete audio;
        delete video; // This also finalizes FORMAT_CONTEXT
    }
};
