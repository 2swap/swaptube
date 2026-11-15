#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "IoHelpers.h"

struct AVCodecContext;
struct AVStream;
struct AVFormatContext;

typedef int32_t sample_t;

const static int audio_channels = 2; // Stereo

inline constexpr sample_t line_max = (static_cast<sample_t>(1) << (sizeof(sample_t) * 8 - 3)) - 1;

inline sample_t float_to_sample(float f) {
    if (f > 1.0f) f = 1.0f;
    if (f < -1.0f) f = -1.0f;
    return static_cast<sample_t>(f * line_max);
}

inline float sample_to_float(sample_t s) {
    return static_cast<float>(s) / line_max;
}

class AudioWriter {
public:
    const bool include_audio;
    const bool audio_sfx;

private:
    const int num_audio_streams;

    std::vector<AVCodecContext*> outputCodecContexts;
    std::vector<AVStream*>      audioStreams;

    AVFormatContext *fc;

    // Interleaved buffers: layout [L0, R0, L1, R1, ...]
    std::vector<sample_t> sample_buffer; // Audio defined in macroblocks (usually voice)
    std::vector<sample_t> sfx_buffer; // Per-scene sound effects

    int total_samples_processed;

    int64_t total_samples_added = 0;

    bool file_exists(const std::string& filename);

    // Encodes and writes packets from a given codec context and stream, returns encoded length in seconds
    void encode_and_write_audio(AVCodecContext* codecCtx, AVStream* stream);

public:
    AudioWriter(AVFormatContext *fc_, int audio_samplerate_hz, const bool& include_audio, const bool& audio_sfx);
    void add_sfx(const std::vector<sample_t>& left_buffer, const std::vector<sample_t>& right_buffer, const double t);

    int64_t add_generated_audio(const std::vector<sample_t>& left_buffer, const std::vector<sample_t>& right_buffer);
    int64_t add_silence(int64_t num_samples);
    int64_t add_audio_from_file(const std::string& filename);

    int64_t get_total_samples_added() const { return total_samples_added; }

    void encode_buffers(bool final_flush = false);

    ~AudioWriter();
};
