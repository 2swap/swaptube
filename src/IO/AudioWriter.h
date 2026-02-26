#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "../Core/State/TransitionType.h"
#include "DebugPlot.h"
#include "IoHelpers.h"

struct AVCodecContext;
struct AVStream;
struct AVFormatContext;

typedef int32_t sample_t;

const static int audio_channels = 2; // Stereo
const static int num_audio_streams = 1 + (AUDIO_SFX?2:0) + (AUDIO_HINTS?2:0);

inline constexpr sample_t line_max = (static_cast<sample_t>(1) << (sizeof(sample_t) * 8 - 3)) - 1;
inline sample_t float_to_sample(float f) {
    if (f > 1.0f) f = 1.0f;
    if (f < -1.0f) f = -1.0f;
    return static_cast<sample_t>(f * line_max);
}

class AudioWriter {
private:
    std::vector<AVCodecContext*> outputCodecContexts;
    std::vector<AVStream*>      audioStreams;

    AVFormatContext *fc;

    // Interleaved buffers: layout [L0, R0, L1, R1, ...]
    std::vector<sample_t> sample_buffer; // Audio defined in macroblocks (usually voice)
    std::vector<sample_t> sfx_buffer; // Per-scene sound effects
    std::vector<sample_t> blips_buffer; // Single-sample blips for audio cues

    int total_samples_processed;

    bool file_exists(const std::string& filename);

    // Encodes and writes packets from a given codec context and stream, returns encoded length in seconds
    void encode_and_write_audio(AVCodecContext* codecCtx, AVStream* stream);

public:
    AudioWriter(AVFormatContext *fc_, int audio_samplerate_hz);
    void add_sfx(const std::vector<sample_t>& left_buffer, const std::vector<sample_t>& right_buffer, const int t);

    // These are used for 6884's transition curve hints
    int current_macroblock_length_samples;
    int current_microblock_length_samples;
    int macroblock_linear_step;
    int microblock_linear_step;

    void add_blip(const int t, const TransitionType tt, const int upcoming_macroblock_length_samples, const int upcoming_microblock_length_samples);
    int add_generated_audio(const std::vector<sample_t>& left_buffer, const std::vector<sample_t>& right_buffer);
    int add_silence(int duration_frames);
    int add_audio_from_file(const std::string& filename);

    int macroblock_line;
    int microblock_line;
    void encode_buffers();

    ~AudioWriter();
};
