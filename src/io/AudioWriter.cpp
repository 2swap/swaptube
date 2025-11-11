#pragma once
#include "ffmpeg_error.hpp"
#include <sys/stat.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <cmath>
#include <cstdint>
#include "DebugPlot.h"
#include "IoHelpers.cpp"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/channel_layout.h>
}

using namespace std;

// Helpful commands
// ffmpeg -h encoder=pcm_s16le # List all supported sample formats for a given encoder
// ffmpeg -encoders | grep pcm # List all relevant wav encoders

// signed integer, non-planar (interleaved)
const AVCodecID output_codec = AV_CODEC_ID_PCM_S32LE;
const AVSampleFormat output_sample_format = AV_SAMPLE_FMT_S32;
typedef int32_t sample_t;
sample_t line_max = (static_cast<sample_t>(1) << (sizeof(sample_t) * 8 - 3)) - 1;
const static int audio_channels = 2; // Stereo
const static int num_audio_streams = 1 + (AUDIO_SFX?2:0) + (AUDIO_HINTS?2:0);

inline sample_t float_to_sample(float f) {
    if (f > 1.0f) f = 1.0f;
    if (f < -1.0f) f = -1.0f;
    return static_cast<sample_t>(f * line_max);
}

class AudioWriter {
private:
    vector<AVCodecContext*> outputCodecContexts = vector<AVCodecContext*>(num_audio_streams, nullptr);
    vector<AVStream*      > audioStreams        = vector<AVStream*      >(num_audio_streams, nullptr);

    AVFormatContext *fc = nullptr;

    // Interleaved buffers: layout [L0, R0, L1, R1, ...]
    vector<sample_t> sample_buffer; // Audio defined in macroblocks (usually voice)
    vector<sample_t> sfx_buffer; // Per-scene sound effects
    vector<sample_t> blips_buffer; // Single-sample blips for audio cues

    int total_samples_processed = 0;

    bool file_exists(const string& filename){
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }

    // Encodes and writes packets from a given codec context and stream, returns encoded length in seconds
    void encode_and_write_audio(AVCodecContext* codecCtx, AVStream* stream) {
        AVPacket outputPacket = {0};

        int ret = 1;
        while (ret >= 0) {
            ret = avcodec_receive_packet(codecCtx, &outputPacket);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }

            outputPacket.stream_index = stream->index;

            pts_dts_plot.add_datapoint(vector<double>{static_cast<double>(outputPacket.pts)});

            av_packet_rescale_ts(&outputPacket, codecCtx->time_base, stream->time_base);

            ret = av_interleaved_write_frame(fc, &outputPacket);
            if (ret < 0) throw runtime_error("Error writing audio frame: " + ff_errstr(ret));

            av_packet_unref(&outputPacket);
        }
    }

public:
    AudioWriter(AVFormatContext *fc_) : fc(fc_), sample_buffer(), sfx_buffer() {
        for(int i = 0; i < num_audio_streams; i++) {
            const AVCodec* audioOutputCodec = avcodec_find_encoder(output_codec);

            if (!audioOutputCodec) throw runtime_error("Error: Could not find audio encoder for codec.");

            outputCodecContexts[i] = avcodec_alloc_context3(audioOutputCodec);
            if (!outputCodecContexts[i]) throw runtime_error("Error: Could not allocate codec context for encoder.");

            outputCodecContexts[i]->sample_rate = SAMPLERATE;
            av_channel_layout_default(&outputCodecContexts[i]->ch_layout, audio_channels);
            outputCodecContexts[i]->sample_fmt = output_sample_format;
            outputCodecContexts[i]->time_base = {1, outputCodecContexts[i]->sample_rate};

            int ret = avcodec_open2(outputCodecContexts[i], audioOutputCodec, nullptr);
            if (ret < 0) {
                avcodec_free_context(&outputCodecContexts[i]);
                throw runtime_error("error: Could not open audio encoder for codec: " + ff_errstr(ret));
            }

            audioStreams[i] = avformat_new_stream(fc, audioOutputCodec);
            if (!audioStreams[i]) {
                avcodec_free_context(&outputCodecContexts[i]);
                throw runtime_error("Error: Could not create new audio stream for codec.");
            }

            audioStreams[i]->time_base = outputCodecContexts[i]->time_base;

            if (avcodec_parameters_from_context(audioStreams[i]->codecpar, outputCodecContexts[i]) < 0) {
                avcodec_free_context(&outputCodecContexts[i]);
                throw runtime_error("Error: Could not initialize stream parameters from codec context.");
            }
        }
    }

    void add_sfx(const vector<sample_t>& left_buffer, const vector<sample_t>& right_buffer, const int t) {
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("SFX buffer lengths do not match. Left: " + to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }

        int numSamples = left_buffer.size(); // number of frames
        int sample_copy_start_frames = t - total_samples_processed;
        int sample_copy_end_frames = sample_copy_start_frames + numSamples;

        if(sample_copy_start_frames < 0)
            throw runtime_error("Sfx copy start was negative: " + to_string(sample_copy_start_frames) + ". " + to_string(t) + " " + to_string(total_samples_processed));

        if (!rendering_on() || !AUDIO_SFX) return; // Don't write in smoketest

        int start_idx = sample_copy_start_frames * audio_channels;
        int end_idx = sample_copy_end_frames * audio_channels;
        // Ensure sfx_buffer has enough capacity and add samples (convert floats to int32)
        if (sfx_buffer.size() < static_cast<size_t>(end_idx)) {
            sfx_buffer.resize(end_idx, 0); // Extend with silence
        }

        for (int i = 0; i < numSamples; ++i) {
            sfx_buffer[start_idx + 2 * i    ] +=  left_buffer[i];
            sfx_buffer[start_idx + 2 * i + 1] += right_buffer[i];
        }
    }

    // These are used for 6884's transition curve hints
    int current_macroblock_length_samples = 0;
    int current_microblock_length_samples = 0;
    int macroblock_linear_step = 0;
    int microblock_linear_step = 0;

    void add_blip(const int t, const TransitionType tt, const int upcoming_macroblock_length_samples, const int upcoming_microblock_length_samples) {
        current_macroblock_length_samples = upcoming_macroblock_length_samples;
        current_microblock_length_samples = upcoming_microblock_length_samples;

        int sample_idx = t - total_samples_processed;

        if(sample_idx < 0)
            throw runtime_error("Blip copy start was negative: " + to_string(sample_idx) + ". " + to_string(t) + " " + to_string(total_samples_processed));

        if (!rendering_on() || !AUDIO_HINTS) return; // Don't write in smoketest

        // Ensure blips_buffer has enough capacity and add samples (convert floats to int32)
        int new_size = (sample_idx + 1) * audio_channels;
        if (blips_buffer.size() < new_size) {
            blips_buffer.resize(new_size); // Extend with silence
        }

        if(tt == MACRO) blips_buffer[sample_idx * 2    ] += 10000000; // Left
        else            blips_buffer[sample_idx * 2 + 1] += 10000000; // Right
    }

    int add_generated_audio(const vector<sample_t>& left_buffer, const vector<sample_t>& right_buffer) {
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("Generated sound buffer lengths do not match. Left: "+ to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }
        if(left_buffer.size() % (SAMPLERATE / FRAMERATE) != 0){
            throw runtime_error("Generated sound buffer length is not a multiple of video frame size. Size: " + to_string(left_buffer.size()) + " samples.");
        }

        if (!rendering_on()) return 0; // Don't write in smoketest

        int num_samples = left_buffer.size();
        for(int i = 0; i < num_samples; i++){
            sample_buffer.push_back( left_buffer[i]);
            sample_buffer.push_back(right_buffer[i]);
        }

        return num_samples * FRAMERATE / SAMPLERATE;
    }

    int add_silence(int duration_frames) {
        if (!rendering_on()) return 0; // Don't write in smoketest
        int num_samples = duration_frames * SAMPLERATE / FRAMERATE;
        sample_buffer.resize(sample_buffer.size() + num_samples * audio_channels, 0);
        return duration_frames;
    }

    int add_audio_from_file(const string& filename) {
        // Build full path to the input audio file
        string fullInputAudioFilename = PATH_MANAGER.this_project_media_dir + filename;

        // Check if the file exists
        if (!file_exists(fullInputAudioFilename)) {
            int seconds = 2;
            cout << "Audio file not found: " << fullInputAudioFilename << ". Adding " << seconds << " seconds of silence instead." << endl;
            return add_silence(seconds * FRAMERATE);
        }

        int ret = 0;

        // Open the input file and its format context
        AVFormatContext* inputAudioFormatContext = nullptr;
        if (ret = avformat_open_input(&inputAudioFormatContext, fullInputAudioFilename.c_str(), nullptr, nullptr) < 0) {
            throw runtime_error("Error: Could not open input audio file: " + fullInputAudioFilename + ". " + ff_errstr(ret));
        }

        // Find stream information
        if (ret = avformat_find_stream_info(inputAudioFormatContext, nullptr) < 0) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Could not retrieve stream info from file: " + fullInputAudioFilename + ". " + ff_errstr(ret));
        }

        // Find the audio stream
        AVStream* audioStreamInput = nullptr;
        for (unsigned int i = 0; i < inputAudioFormatContext->nb_streams; ++i) {
            if (inputAudioFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamInput = inputAudioFormatContext->streams[i];
                break;
            }
        }
        if (!audioStreamInput) {
            throw runtime_error("Error: No audio stream found in file: " + fullInputAudioFilename);
        }

        if (audioStreamInput->codecpar->codec_id != output_codec) {
            throw runtime_error("Error: Input file is not in expected format: " + fullInputAudioFilename);
        }

        if (audioStreamInput->codecpar->format != output_sample_format) {
            throw runtime_error("Error: Input file is not in expected sample format: " + fullInputAudioFilename);
        }

        // Check sample rate
        if (audioStreamInput->codecpar->sample_rate != SAMPLERATE) {
            throw runtime_error("Error: Unsupported sample rate: " + to_string(audioStreamInput->codecpar->sample_rate) + ". Expected " + to_string(SAMPLERATE) + " Hz.");
        }

        // Ensure the audio is stereo
        if (audioStreamInput->codecpar->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
            throw runtime_error("Error: Channel order is unspecified.");
        }
        int num_channels = audioStreamInput->codecpar->ch_layout.nb_channels;
        if (num_channels != audio_channels) {
            throw runtime_error("Error: Unsupported channel count: " + to_string(num_channels) + ". Expected " + to_string(audio_channels) + " channels.");
        }

        // Set up the audio decoder
        const AVCodec* audioDecoder = avcodec_find_decoder(output_codec);
        if (!audioDecoder) {
            throw runtime_error("Error: appropriate audio decoder not found.");
        }

        AVCodecContext* codecContext = avcodec_alloc_context3(audioDecoder);
        if (!codecContext) {
            throw runtime_error("Error: Could not allocate codec context for audio decoder.");
        }

        if (ret = avcodec_parameters_to_context(codecContext, audioStreamInput->codecpar) < 0) {
            throw runtime_error("Error: Could not initialize codec context from stream parameters. " + ff_errstr(ret));
        }

        if (ret = avcodec_open2(codecContext, audioDecoder, nullptr) < 0) {
            throw runtime_error("Error: Could not open audio decoder. " + ff_errstr(ret));
        }

        if (!rendering_on()) {
            avcodec_free_context(&codecContext);
            avformat_close_input(&inputAudioFormatContext);
            return 0; // Don't write in smoketest
        }

        // Decode and process the audio samples
        AVPacket* packet = av_packet_alloc();
        if(!packet){
            throw runtime_error("Error: Failed to allocate AVPacket.");
        }

        int length_in_samples = 0;
        while (av_read_frame(inputAudioFormatContext, packet) >= 0) {
            if (packet->stream_index == audioStreamInput->index) {
                AVFrame* frame = av_frame_alloc();
                if (!frame) {
                    throw runtime_error("Error: Could not allocate frame.");
                }

                int ret = avcodec_send_packet(codecContext, packet);
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codecContext, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    }

                    int num_samples = frame->nb_samples;

                    for (int i = 0; i < num_samples; ++i) {
                        sample_buffer.push_back(reinterpret_cast<sample_t*>(frame->data[0])[i*2]); // Left
                        sample_buffer.push_back(reinterpret_cast<sample_t*>(frame->data[0])[i*2+1]); // Right
                    }

                    length_in_samples += num_samples;

                    av_frame_unref(frame);
                }

                av_frame_free(&frame);
            }
            av_packet_unref(packet);
        }

        // Add silence to align to a frame boundary
        int SAMPLES_PER_VIDEO_FRAME = SAMPLERATE / FRAMERATE;
        while (length_in_samples % SAMPLES_PER_VIDEO_FRAME != 0) {
            sample_buffer.push_back(0);
            sample_buffer.push_back(0);
            length_in_samples++;
        }

        avcodec_free_context(&codecContext);
        avformat_close_input(&inputAudioFormatContext);

        return length_in_samples / SAMPLES_PER_VIDEO_FRAME;
    }

    int macroblock_line = 0;
    int microblock_line = 0;
    void encode_buffers() {
        if(!rendering_on()) return; // Don't write in smoketest

        while(true){
            int frameSize = outputCodecContexts[0]->frame_size;
            if(frameSize <= 0) frameSize = 1024; // Default frame size if not set

            // If we don't have enough samples for a full frame, exit
            if(sample_buffer.size() < frameSize * audio_channels) break;
            // TODO pad last frame?

            for (int ch = 0; ch < audio_channels; ++ch) {
                // Extend channel with silence (interleaved)
                if (  sfx_buffer.size() < sample_buffer.size())   sfx_buffer.resize(sample_buffer.size(), 0);
                if (blips_buffer.size() < sample_buffer.size()) blips_buffer.resize(sample_buffer.size(), 0);
            }

            vector<AVFrame*> frames = vector<AVFrame*>(num_audio_streams, nullptr);
            for(int i = 0; i < num_audio_streams; i++) {
                frames[i] = av_frame_alloc();
                if(!frames[i]) throw runtime_error("Frame allocation failed.");
            }

            // Setup frame properties
            auto setupFrame = [&](AVFrame* frame, AVCodecContext* codecCtx){
                frame->nb_samples = frameSize;
                av_channel_layout_default(&frame->ch_layout, 2); // stereo
                frame->sample_rate = codecCtx->sample_rate;
                frame->format = codecCtx->sample_fmt;

                // Allocate buffer for the frame
                int ret = av_frame_get_buffer(frame, 0);
                if (ret < 0) throw runtime_error("Error allocating audio frame buffer: " + ff_errstr(ret));
            };

            for(int i = 0; i < num_audio_streams; i++) {
                setupFrame(frames[i], outputCodecContexts[i]);
            }

            vector<sample_t*> dst = vector<sample_t*>(num_audio_streams, nullptr);
            for(int i = 0; i < num_audio_streams; i++) {
                dst[i] = reinterpret_cast<sample_t*>(frames[i]->data[0]);
            }

            for (int i = 0; i < frameSize; ++i) {
                int idxL = 2 * i;
                int idxR = 2 * i + 1;

                sample_t voice_left = sample_buffer[idxL];
                sample_t voice_right = sample_buffer[idxR];
                sample_t sfx_left = sfx_buffer[idxL];
                sample_t sfx_right = sfx_buffer[idxR];
                sample_t blips_left = blips_buffer[idxL];
                sample_t blips_right = blips_buffer[idxR];

                int track_number = 0;

                if(AUDIO_SFX){
                    // Merged audio track
                    dst[track_number][idxL] = voice_left + sfx_left;
                    dst[track_number][idxR] = voice_right+ sfx_right;
                    track_number++;

                    // Sfx-only track
                    dst[track_number][idxL] = sfx_left;
                    dst[track_number][idxR] = sfx_right;
                    track_number++;
                }

                // Always include voice.
                dst[track_number][idxL] = voice_left;
                dst[track_number][idxR] = voice_right;
                track_number++;

                if(AUDIO_HINTS){
                    // Blips-only track
                    dst[track_number][idxL] = blips_left;
                    dst[track_number][idxR] = blips_right;
                    track_number++;

                    if(blips_left != 0){ // All microblocks are same length, so only reset on macroblock
                        macroblock_linear_step = line_max / current_macroblock_length_samples;
                        microblock_linear_step = line_max / current_microblock_length_samples;
                        macroblock_line = 0;
                    }
                    if(blips_right != 0){ // Right is microblock
                        microblock_line = 0;
                    }

                    // Transition Lines
                    dst[track_number][idxL] = macroblock_line;
                    dst[track_number][idxR] = microblock_line;
                    track_number++;

                    macroblock_line += macroblock_linear_step;
                    microblock_line += microblock_linear_step;
                }
            }

            for(int i = 0; i < num_audio_streams; i++) {
                frames[i]->pts = total_samples_processed;
                int ret = avcodec_send_frame(outputCodecContexts[i], frames[i]);
                av_frame_free(&frames[i]);
                if (ret < 0)
                    throw runtime_error("Error sending audio frame to encoder " + to_string(i) + ": " + ff_errstr(ret));
                encode_and_write_audio(outputCodecContexts[i], audioStreams[i]);
            }

            // Erase the samples used in this frame (interleaved count)
            sample_buffer.erase(sample_buffer.begin(), sample_buffer.begin() + frameSize * audio_channels);
               sfx_buffer.erase(   sfx_buffer.begin(),    sfx_buffer.begin() + frameSize * audio_channels);
             blips_buffer.erase( blips_buffer.begin(),  blips_buffer.begin() + frameSize * audio_channels);
            total_samples_processed += frameSize;
        }
    }

    ~AudioWriter() {
        cout << "Cleaning up AudioWriter..." << endl;
        encode_buffers();

        // Flush encoders by sending null frames
        for(int i = 0; i < num_audio_streams; i++) {
            avcodec_send_frame(outputCodecContexts[i], NULL);
            encode_and_write_audio(outputCodecContexts[i], audioStreams[i]);
            avcodec_free_context(&outputCodecContexts[i]);
        }
    }
};
