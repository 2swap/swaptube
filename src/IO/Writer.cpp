#include <stdexcept>
#include <memory>
#include <cstdint>
#include <string>
#include "Writer.h"
#include "ShtookaWriter.h"
#include "SubtitleWriter.h"
#include "AudioWriter.h"
#include "VideoWriter.h"
#include "../Core/Smoketest.h"

using namespace std;

Writer::Writer(int video_width_pixels, int video_height_pixels, int video_framerate_fps, int audio_samplerate_hz, uint32_t video_background_color) :
    video_width_pixels(video_width_pixels),
    video_height_pixels(video_height_pixels),
    video_framerate_fps(video_framerate_fps),
    audio_samplerate_hz(audio_samplerate_hz),
    video_background_color(video_background_color)
{
    shtooka = new ShtookaWriter();
    subtitle = new SubtitleWriter();

    if (is_smoketest()) return;
    const std::string video_path = "io_out/Video.mkv";
    int ret = avformat_alloc_output_context2(&format_context, NULL, NULL, video_path.c_str());
    if (ret < 0) throw std::runtime_error("Failed to allocate output format context");
    if (format_context == nullptr) throw std::runtime_error("Failed to allocate output format context");

    audio = new AudioWriter(format_context, audio_samplerate_hz);
    video = new VideoWriter(format_context, video_path, video_width_pixels, video_height_pixels, video_framerate_fps);
}

Writer::~Writer() {
    delete shtooka;
    delete subtitle;

    if (is_smoketest()) return;
    delete audio;
    delete video; // This also finalizes FORMAT_CONTEXT
}

int Writer::get_video_width_pixels() const { return video_width_pixels; }
int Writer::get_video_height_pixels() const { return video_height_pixels; }
int Writer::get_video_framerate_fps() const { return video_framerate_fps; }
int Writer::get_audio_samplerate_hz() const { return audio_samplerate_hz; }
uint32_t Writer::get_video_background_color() const { return video_background_color; }

static std::unique_ptr<Writer> writer;

void init_writer(int video_width_pixels, int video_height_pixels, int video_framerate_fps, int audio_samplerate_hz, uint32_t video_background_color) {
    cout << "Initializing writer... " << flush;
    if (writer)
        throw std::runtime_error("Writer already initialized");

    writer = std::make_unique<Writer>(video_width_pixels, video_height_pixels, video_framerate_fps, audio_samplerate_hz, video_background_color);
    cout << "Done." << endl;
}

Writer& get_writer() {
    if (!writer)
        throw std::runtime_error("Writer not initialized");
    return *writer;
}

int get_audio_samplerate_hz() {
    return get_writer().get_audio_samplerate_hz();
}
int get_video_width_pixels() {
    return get_writer().get_video_width_pixels();
}
int get_video_framerate_fps() {
    return get_writer().get_video_framerate_fps();
}
int get_video_height_pixels() {
    return get_writer().get_video_height_pixels();
}
uint32_t get_video_background_color() {
    return get_writer().get_video_background_color();
}
int get_samples_per_frame() {
    return get_writer().get_audio_samplerate_hz() / get_writer().get_video_framerate_fps();
}
float get_video_aspect_ratio() {
    return get_writer().get_video_width_pixels() / get_writer().get_video_height_pixels();
}
