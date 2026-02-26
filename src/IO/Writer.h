#pragma once

#include <string>
#include <stdexcept>

#include "AudioWriter.h"
#include "VideoWriter.h"
#include "SubtitleWriter.h"
#include "ShtookaWriter.h"

class Writer {
public:
    AVFormatContext* format_context = nullptr;
    AudioWriter* audio = nullptr;
    VideoWriter* video = nullptr;
    SubtitleWriter* subtitle = nullptr;
    ShtookaWriter* shtooka = nullptr;

    Writer(int video_width_pixels, int video_height_pixels, int video_framerate_fps, int audio_samplerate_hz, uint32_t video_background_color);
    ~Writer();

    int get_video_width_pixels() const;
    int get_video_height_pixels() const;
    int get_video_framerate_fps() const;
    int get_audio_samplerate_hz() const;
    uint32_t get_video_background_color() const;
private:
    const int video_width_pixels;
    const int video_height_pixels;
    const int video_framerate_fps;
    const int audio_samplerate_hz;
    const uint32_t video_background_color = 0x00000000;
};

void init_writer(int video_width_pixels, int video_height_pixels, int video_framerate_fps, int audio_samplerate_hz, uint32_t video_background_color);
Writer& get_writer();

int get_video_width_pixels();
int get_video_height_pixels();
float get_video_aspect_ratio();
int get_video_framerate_fps();
int get_audio_samplerate_hz();
uint32_t get_video_background_color();
int get_samples_per_frame();
