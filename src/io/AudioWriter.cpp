#pragma once
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


class AudioWriter {
private:
    // Codec contexts for three audio streams
    AVCodecContext *audioOutputCodecContextMerged = nullptr;
    AVCodecContext *audioOutputCodecContextSFX = nullptr;
    AVCodecContext *audioOutputCodecContextVoice = nullptr;

    AVStream *audioStreamMerged = nullptr;
    AVStream *audioStreamSFX = nullptr;
    AVStream *audioStreamVoice = nullptr;

    AVFormatContext *fc = nullptr;

    // Interleaved 32-bit signed integer buffers: layout [L0, R0, L1, R1, ...]
    vector<int32_t> sample_buffer;
    vector<int32_t> sfx_buffer;

    int sample_buffer_offset = 0; // The index in the sample_buffer at which the latest macroblock starts, since some samples from the last one may not have flushed entirely.
    int total_samples_processed = 0;

    bool file_exists(const string& filename){
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }

    // Encodes and writes packets from a given codec context and stream, returns encoded length in seconds
    double encode_and_write_audio(AVCodecContext* codecCtx, AVStream* stream) {
        AVPacket outputPacket = {0};

        double length_in_seconds = 0;

        int ret = 1;
        while (ret >= 0) {
            ret = avcodec_receive_packet(codecCtx, &outputPacket);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }

            outputPacket.stream_index = stream->index;

            pts_dts_plot.add_datapoint(vector<double>{static_cast<double>(outputPacket.pts)});
            length_in_seconds += static_cast<double>(outputPacket.duration) / codecCtx->sample_rate;

            av_packet_rescale_ts(&outputPacket, codecCtx->time_base, stream->time_base);

            ret = av_interleaved_write_frame(fc, &outputPacket);

            av_packet_unref(&outputPacket);
        }
        return length_in_seconds;
    }

    // Encodes and writes audio for all three streams, returns the maximum length among them
    void encode_and_write_all() {
        double lengthMerged = encode_and_write_audio(audioOutputCodecContextMerged, audioStreamMerged);
        double lengthSFX = encode_and_write_audio(audioOutputCodecContextSFX, audioStreamSFX);
        double lengthVoice = encode_and_write_audio(audioOutputCodecContextVoice, audioStreamVoice);
    }

public:
    double audio_seconds_so_far = 0;

    AudioWriter(AVFormatContext *fc_) : fc(fc_), sample_buffer(), sfx_buffer() {
        // Helper lambda to setup codec context and stream for encoder
        auto setup_stream = [this](AVFormatContext* fc, AVCodecContext*& codecCtx, AVStream*& stream, const char* codec_name) {
            const AVCodec* audioOutputCodec = avcodec_find_encoder(AV_CODEC_ID_PCM_S32LE); // Use 32-bit signed integer, non-planar (interleaved)

            /*
            cout << "Supported sample formats for encoder " << codec_name << ":" << endl;
            for (int i = 0; audioOutputCodec->sample_fmts; i++){
                const char* sample_fmt_name = av_get_sample_fmt_name(audioOutputCodec->sample_fmts[i]);
                printf("  %s\n", sample_fmt_name ? sample_fmt_name : "unknown");
                if(audioOutputCodec->sample_fmts[i] == AV_SAMPLE_FMT_NONE) break;
            }
            */

            if (!audioOutputCodec) throw runtime_error(string("Error: Could not find audio encoder for ") + codec_name + ".");

            codecCtx = avcodec_alloc_context3(audioOutputCodec);
            if (!codecCtx) throw runtime_error(string("Error: Could not allocate codec context for encoder ") + codec_name + ".");

            codecCtx->sample_rate = SAMPLERATE;
            av_channel_layout_default(&codecCtx->ch_layout, 2); // stereo
            codecCtx->sample_fmt = AV_SAMPLE_FMT_S32; // Use 32-bit signed integer, packed (non-planar)
            codecCtx->time_base = {1, codecCtx->sample_rate}; // 1/sample_rate

            int ret = avcodec_open2(codecCtx, audioOutputCodec, nullptr);
            if (ret < 0) {
                cout << "avcodec_open2 error: " << av_err2str(ret) << endl;
                avcodec_free_context(&codecCtx);
                throw runtime_error(string("Error: Could not open encoder ") + codec_name + ".");
            }

            stream = avformat_new_stream(fc, audioOutputCodec);
            if (!stream) {
                avcodec_free_context(&codecCtx);
                throw runtime_error(string("Error: Could not create audio stream in output format context for ") + codec_name + ".");
            }

            stream->time_base = codecCtx->time_base;

            if (avcodec_parameters_from_context(stream->codecpar, codecCtx) < 0) {
                avcodec_free_context(&codecCtx);
                throw runtime_error(string("Error: Could not initialize stream codec parameters for ") + codec_name + ".");
            }
        };

        cout << "Setting up audio streams..." << endl;
        setup_stream(fc, audioOutputCodecContextMerged, audioStreamMerged, "Merged");
        setup_stream(fc, audioOutputCodecContextSFX   , audioStreamSFX   , "SFX"   );
        setup_stream(fc, audioOutputCodecContextVoice , audioStreamVoice , "Voice" );
        cout << "Audio streams setup complete." << endl;
    }

    void add_sfx(const vector<int32_t>& left_buffer, const vector<int32_t>& right_buffer, int t) {
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("SFX buffer lengths do not match. Left: " + to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }

        int numSamples = left_buffer.size(); // number of frames
        int channels = 2;
        int sample_copy_start_frames = t - total_samples_processed + sample_buffer_offset;
        int sample_copy_end_frames = sample_copy_start_frames + numSamples;

        if(sample_copy_start_frames < 0)
            throw runtime_error("Sample copy start was negative: " + to_string(sample_copy_start_frames) + ". " + to_string(t) + " " + to_string(total_samples_processed) + " " + to_string(sample_buffer_offset));

        int start_idx = sample_copy_start_frames * channels;
        int end_idx = sample_copy_end_frames * channels;

        // Ensure sfx_buffer has enough capacity and add samples (convert floats to int32)
        if (sfx_buffer.size() < static_cast<size_t>(end_idx)) {
            sfx_buffer.resize(end_idx, 0); // Extend with silence
        }

        for (int i = 0; i < numSamples; ++i) {
            sfx_buffer[start_idx + 2 * i    ] +=  left_buffer[i];
            sfx_buffer[start_idx + 2 * i + 1] += right_buffer[i];
        }
    }

    double add_generated_audio(const vector<int32_t>& left_buffer, const vector<int32_t>& right_buffer) {
        process_frame_from_buffer();
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("Generated sound buffer lengths do not match. Left: "+ to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }

        int num_samples = left_buffer.size();

        for(int i = 0; i < num_samples; i++){
            sample_buffer.push_back( left_buffer[i]);
            sample_buffer.push_back(right_buffer[i]);
        }

        return static_cast<double>(num_samples) / audioOutputCodecContextMerged->sample_rate; // sample_rate same for all
    }

    void add_silence(double duration) {
        process_frame_from_buffer();
        int num_samples = static_cast<int>(duration * audioOutputCodecContextMerged->sample_rate);

        for (int i = 0; i < num_samples; ++i) {
            sample_buffer.push_back(0); // L
            sample_buffer.push_back(0); // R
        }
    }

    double add_audio_from_file(const string& filename) {
        process_frame_from_buffer();
        double length_in_seconds = 0;

        // Build full path to the input audio file
        string fullInputAudioFilename = PATH_MANAGER.this_project_media_dir + filename;

        // Check if the file exists
        if (!file_exists(fullInputAudioFilename)) {
            int seconds = 2;
            cout << "Audio file not found: " << fullInputAudioFilename << ". Adding " << seconds << " seconds of silence instead." << endl;
            add_silence(seconds);
            return seconds;
        }

        // Open the input file and its format context
        AVFormatContext* inputAudioFormatContext = nullptr;
        if (avformat_open_input(&inputAudioFormatContext, fullInputAudioFilename.c_str(), nullptr, nullptr) < 0) {
            throw runtime_error("Error: Could not open input audio file: " + fullInputAudioFilename);
        }

        // Find stream information
        if (avformat_find_stream_info(inputAudioFormatContext, nullptr) < 0) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Could not retrieve stream info from file: " + fullInputAudioFilename);
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

        if (audioStreamInput->codecpar->codec_id != AV_CODEC_ID_PCM_S32LE) {
            throw runtime_error("Error: Input file is not in expected format: " + fullInputAudioFilename);
        }

        if (audioStreamInput->codecpar->format != AV_SAMPLE_FMT_S32) {
            throw runtime_error("Error: Input file is not in expected sample format: " + fullInputAudioFilename);
        }

        // Check sample rate
        if (audioStreamInput->codecpar->sample_rate != SAMPLERATE) {
            throw runtime_error("Error: Unsupported sample rate: " + to_string(audioStreamInput->codecpar->sample_rate) + ". Expected " + to_string(SAMPLERATE) + " Hz.");
        }

        // Ensure the audio is stereo
        int num_channels = 0;
        if (audioStreamInput->codecpar->ch_layout.order != AV_CHANNEL_ORDER_UNSPEC) {
            num_channels = audioStreamInput->codecpar->ch_layout.nb_channels;
        } else {
            throw runtime_error("Error: Channel order is unspecified.");
        }
        if (num_channels != 2) {
            throw runtime_error("Error: Unsupported channel count: " + to_string(num_channels) + ". Expected stereo (2 channels).");
        }

        // Set up the audio decoder
        const AVCodec* audioDecoder = avcodec_find_decoder(AV_CODEC_ID_PCM_S32LE);
        if (!audioDecoder) {
            throw runtime_error("Error: appropriate audio decoder not found.");
        }

        AVCodecContext* codecContext = avcodec_alloc_context3(audioDecoder);
        if (!codecContext) {
            throw runtime_error("Error: Could not allocate codec context for audio decoder.");
        }

        if (avcodec_parameters_to_context(codecContext, audioStreamInput->codecpar) < 0) {
            throw runtime_error("Error: Could not initialize codec context from stream parameters.");
        }

        if (avcodec_open2(codecContext, audioDecoder, nullptr) < 0) {
            throw runtime_error("Error: Could not open audio decoder.");
        }

        // Decode and process the audio samples
        AVPacket* packet = av_packet_alloc();
        if(!packet){
            throw runtime_error("Error: Failed to allocate AVPacket.");
        }
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
                        sample_buffer.push_back(reinterpret_cast<int32_t*>(frame->data[0])[i*2]); // Left
                        sample_buffer.push_back(reinterpret_cast<int32_t*>(frame->data[0])[i*2+1]); // Right
                    }

                    length_in_seconds += static_cast<double>(num_samples) / codecContext->sample_rate;

                    av_frame_unref(frame);
                }

                av_frame_free(&frame);
            }
            av_packet_unref(packet);
        }

        avcodec_free_context(&codecContext);
        avformat_close_input(&inputAudioFormatContext);

        return length_in_seconds;
    }

    void process_frame_from_buffer(const bool last = false) {
        while(true){
            int channels = audioOutputCodecContextMerged->ch_layout.nb_channels;
            int frameSize = audioOutputCodecContextMerged->frame_size;
            if(frameSize <= 0) {
                frameSize = 1024; // Default frame size if not set
            }

            if (channels != 2) throw runtime_error("Expected stereo channels (2).");
            if(sample_buffer.size() % channels != 0) throw runtime_error("Voice buffer interleaved size mismatch!");
            if(sfx_buffer.size() % channels != 0) throw runtime_error("SFX buffer interleaved size mismatch!");
            if(sample_buffer.size() < frameSize * channels * (last?1:2)) break;

            for (int ch = 0; ch < channels; ++ch) {
                if (sfx_buffer.size() < sample_buffer.size()) {
                    sfx_buffer.resize(sample_buffer.size(), 0); // Extend channel with silence (interleaved)
                }
            }

            AVFrame* frameMerged = av_frame_alloc();
            AVFrame* frameSFX = av_frame_alloc();
            AVFrame* frameVoice = av_frame_alloc();

            if (!frameMerged || !frameSFX || !frameVoice) {
                if(frameMerged) av_frame_free(&frameMerged);
                if(frameSFX) av_frame_free(&frameSFX);
                if(frameVoice) av_frame_free(&frameVoice);
                throw runtime_error("Frame allocation failed.");
            }

            // Setup frame properties
            auto setupFrame = [&](AVFrame* frame, AVCodecContext* codecCtx){
                frame->nb_samples = frameSize;
                av_channel_layout_default(&frame->ch_layout, 2); // stereo
                frame->sample_rate = codecCtx->sample_rate;
                frame->format = codecCtx->sample_fmt;

                // Allocate buffer for the frame
                int ret = av_frame_get_buffer(frame, 0);
                if (ret < 0) {
                    av_frame_free(&frameMerged);
                    av_frame_free(&frameSFX);
                    av_frame_free(&frameVoice);
                    throw runtime_error("Error allocating audio frame buffer: " + string(av_err2str(ret)));
                }
            };

            setupFrame(frameMerged, audioOutputCodecContextMerged);
            setupFrame(frameSFX, audioOutputCodecContextSFX);
            setupFrame(frameVoice, audioOutputCodecContextVoice);

            // Fill buffers (interleaved)
            int required_samples = frameSize * channels;
            // Ensure bounds for `sample_buffer`, `sfx_buffer`, and `frame->data[ch]`
            if (sample_buffer.size() < required_samples || sfx_buffer.size() < required_samples) {
                throw runtime_error("Audio buffer size is smaller than the required frame size.");
            }

            int32_t* dstMerged = reinterpret_cast<int32_t*>(frameMerged->data[0]);
            int32_t* dstSFX = reinterpret_cast<int32_t*>(frameSFX->data[0]);
            int32_t* dstVoice = reinterpret_cast<int32_t*>(frameVoice->data[0]);

            for (int i = 0; i < frameSize; ++i) {
                int idxL = 2 * i;
                int idxR = 2 * i + 1;

                int32_t voice_left = sample_buffer[idxL];
                int32_t voice_right = sample_buffer[idxR];
                int32_t sfx_left = sfx_buffer[idxL];
                int32_t sfx_right = sfx_buffer[idxR];

                dstMerged[idxL] = voice_left + sfx_left;
                dstMerged[idxR] = voice_right+ sfx_right;

                dstSFX[idxL] = sfx_left;
                dstSFX[idxR] = sfx_right;

                dstVoice[idxL] = voice_left;
                dstVoice[idxR] = voice_right;
            }

            frameMerged->pts = total_samples_processed;
            frameSFX  ->pts = total_samples_processed;
            frameVoice->pts = total_samples_processed;

            // Send frames to corresponding encoders
            int retMerged = avcodec_send_frame(audioOutputCodecContextMerged, frameMerged);
            int retSFX = avcodec_send_frame(audioOutputCodecContextSFX, frameSFX);
            int retVoice = avcodec_send_frame(audioOutputCodecContextVoice, frameVoice);

            av_frame_free(&frameMerged);
            av_frame_free(&frameSFX);
            av_frame_free(&frameVoice);

            if (retMerged < 0) throw runtime_error("Error sending merged frame to encoder.");
            if (retSFX < 0) throw runtime_error("Error sending sfx frame to encoder.");
            if (retVoice < 0) throw runtime_error("Error sending voice frame to encoder.");

            encode_and_write_all();

            // Erase the samples used in this frame (interleaved count)
            sample_buffer.erase(sample_buffer.begin(), sample_buffer.begin() + required_samples);
               sfx_buffer.erase(   sfx_buffer.begin(),    sfx_buffer.begin() + required_samples);
            total_samples_processed += frameSize;
        }

        sample_buffer_offset = static_cast<int>(sample_buffer.size() / 2); // number of remaining frames
    }

    ~AudioWriter() {
        process_frame_from_buffer(true);

        // Flush encoders by sending null frames
        avcodec_send_frame(audioOutputCodecContextMerged, NULL);
        avcodec_send_frame(audioOutputCodecContextSFX, NULL);
        avcodec_send_frame(audioOutputCodecContextVoice, NULL);

        encode_and_write_all();

        avcodec_free_context(&audioOutputCodecContextMerged);
        avcodec_free_context(&audioOutputCodecContextSFX);
        avcodec_free_context(&audioOutputCodecContextVoice);
    }
};
