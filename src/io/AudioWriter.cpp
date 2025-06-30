#pragma once
#include <sys/stat.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
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

    vector<vector<float>> sample_buffer;
    vector<vector<float>> sfx_buffer;

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

    AudioWriter(AVFormatContext *fc_) : fc(fc_), sample_buffer(2), sfx_buffer(2) {
        // Helper lambda to setup codec context and stream for encoder
        auto setup_stream = [this](AVFormatContext* fc, AVCodecContext*& codecCtx, AVStream*& stream, const char* codec_name) {
            const AVCodec* audioOutputCodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
            if (!audioOutputCodec) throw runtime_error(string("Error: Could not find audio encoder for ") + codec_name + ".");

            codecCtx = avcodec_alloc_context3(audioOutputCodec);
            if (!codecCtx) throw runtime_error(string("Error: Could not allocate codec context for encoder ") + codec_name + ".");

            codecCtx->bit_rate = 128000; // 128 kbps
            codecCtx->sample_rate = SAMPLERATE;
            av_channel_layout_default(&codecCtx->ch_layout, 2); // stereo
            codecCtx->sample_fmt = AV_SAMPLE_FMT_FLTP; // Floating-point planar format
            codecCtx->time_base = {1, codecCtx->sample_rate}; // 1/sample_rate
            codecCtx->profile = FF_PROFILE_AAC_LOW; // Low complexity AAC (LC-AAC)

            if (avcodec_open2(codecCtx, audioOutputCodec, nullptr) < 0) {
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

        setup_stream(fc, audioOutputCodecContextMerged, audioStreamMerged, "Merged");
        setup_stream(fc, audioOutputCodecContextSFX   , audioStreamSFX   , "SFX"   );
        setup_stream(fc, audioOutputCodecContextVoice , audioStreamVoice , "Voice" );
    }

    void add_sfx(const vector<float>& left_buffer, const vector<float>& right_buffer, int t) {
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("SFX buffer lengths do not match. Left: " + to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }

        int numSamples = left_buffer.size();
        int sample_copy_start = t - total_samples_processed + sample_buffer_offset;
        int sample_copy_end = sample_copy_start + numSamples;

        if(sample_copy_start < 0)
            throw runtime_error("Sample copy start was negative: " + to_string(sample_copy_start) + ". " + to_string(t) + " " + to_string(total_samples_processed) + " " + to_string(sample_buffer_offset));

        // Pointers to the input buffers for each channel
        const vector<float>* input_buffers[2] = {&left_buffer, &right_buffer};

        // Ensure sfx_buffer has enough capacity and add samples
        for (int ch = 0; ch < 2; ++ch) {
            if (sfx_buffer[ch].size() < sample_copy_end) {
                sfx_buffer[ch].resize(sample_copy_end, 0.0f); // Extend channel with silence
            }

            for (int i = 0; i < numSamples; ++i) {
                sfx_buffer[ch][i + sample_copy_start] += (*input_buffers[ch])[i];
            }
        }
    }

    double add_generated_audio(const vector<float>& left_buffer, const vector<float>& right_buffer) {
        process_frame_from_buffer();
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("Generated sound buffer lengths do not match. Left: "+ to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }

        int num_samples = left_buffer.size();

        for(int i = 0; i < num_samples; i++){
            sample_buffer[0].push_back( left_buffer[i]);
            sample_buffer[1].push_back(right_buffer[i]);
        }

        return static_cast<double>(num_samples) / audioOutputCodecContextMerged->sample_rate; // sample_rate same for all
    }

    void add_silence(double duration) {
        process_frame_from_buffer();
        int num_samples = static_cast<int>(duration * audioOutputCodecContextMerged->sample_rate);

        for (int i = 0; i < num_samples; ++i) {
            sample_buffer[0].push_back(0);
            sample_buffer[1].push_back(0);
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

        // Find the audio stream and ensure it is AAC
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

        if (audioStreamInput->codecpar->codec_id != AV_CODEC_ID_AAC) {
            throw runtime_error("Error: Input file is not in expected format: " + fullInputAudioFilename);
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
        const AVCodec* audioDecoder = avcodec_find_decoder(AV_CODEC_ID_AAC);
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
                        sample_buffer[0].push_back(reinterpret_cast<float*>(frame->data[0])[i]); // Left channel
                        sample_buffer[1].push_back(reinterpret_cast<float*>(frame->data[1])[i]); // Right channel
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

            if(sample_buffer[0].size() != sample_buffer[1].size()) throw runtime_error("Voice buffer planar size mismatch!");
            if(sfx_buffer[0].size() != sfx_buffer[1].size()) throw runtime_error("SFX buffer planar size mismatch!");
            if(sample_buffer[0].size() < frameSize*(last?1:2)) break;

            for (int ch = 0; ch < channels; ++ch) {
                if (sfx_buffer[ch].size() < sample_buffer[ch].size()) {
                    sfx_buffer[ch].resize(sample_buffer[ch].size(), 0.0f); // Extend channel with silence
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
                    throw runtime_error("Error allocating audio frame buffer.");
                }
            };

            setupFrame(frameMerged, audioOutputCodecContextMerged);
            setupFrame(frameSFX, audioOutputCodecContextSFX);
            setupFrame(frameVoice, audioOutputCodecContextVoice);

            // Fill buffers
            for (int ch = 0; ch < channels; ++ch) {
                float* dstMerged = reinterpret_cast<float*>(frameMerged->data[ch]);
                float* dstSFX = reinterpret_cast<float*>(frameSFX->data[ch]);
                float* dstVoice = reinterpret_cast<float*>(frameVoice->data[ch]);
                // Ensure bounds for `sample_buffer`, `sfx_buffer`, and `frame->data[ch]`
                if (sample_buffer[ch].size() < frameSize || sfx_buffer[ch].size() < frameSize) {
                    throw runtime_error("Audio buffer size is smaller than the required frame size.");
                }

                for (int i = 0; i < frameSize; ++i) {
                    float voice_sample = sample_buffer[ch][i];
                    float sfx_sample = sfx_buffer[ch][i];
                    float merged_sample = voice_sample + sfx_sample; // Perform element-wise addition

                    if(isnan(merged_sample) || isinf(merged_sample)) {
                        throw runtime_error("Merged audio was either inf or nan. Voice: " + to_string(voice_sample) + ", SFX: " + to_string(sfx_sample));
                    }

                    dstMerged[i] = merged_sample;
                    dstSFX[i] = sfx_sample;
                    dstVoice[i] = voice_sample;
                }
            }

            frameMerged->pts = total_samples_processed;
            frameSFX  ->pts = total_samples_processed;
            frameVoice->pts = total_samples_processed;

            // Send frames to corresponding encoders
            int retMerged = avcodec_send_frame(audioOutputCodecContextMerged, frameMerged);
            int retSFX = avcodec_send_frame(audioOutputCodecContextSFX, frameSFX);
            int retVoice = avcodec_send_frame(audioOutputCodecContextVoice, frameVoice);

            //av_frame_unref(frame);
            av_frame_free(&frameMerged);
            av_frame_free(&frameSFX);
            av_frame_free(&frameVoice);

            if (retMerged < 0) throw runtime_error("Error sending merged frame to encoder.");
            if (retSFX < 0) throw runtime_error("Error sending sfx frame to encoder.");
            if (retVoice < 0) throw runtime_error("Error sending voice frame to encoder.");

            encode_and_write_all();

            // Erase the samples used in this frame
            for (int ch = 0; ch < channels; ++ch) {
                sample_buffer[ch].erase(sample_buffer[ch].begin(), sample_buffer[ch].begin() + frameSize);
                   sfx_buffer[ch].erase(   sfx_buffer[ch].begin(),    sfx_buffer[ch].begin() + frameSize);
            }
            total_samples_processed += frameSize;
        }

        sample_buffer_offset = static_cast<int>(sample_buffer[0].size());
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
