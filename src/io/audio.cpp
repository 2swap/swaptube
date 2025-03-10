#pragma once
#include <sys/stat.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include "AudioSegment.cpp"
#include "DebugPlot.h"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/channel_layout.h>
}

using namespace std;


class AudioWriter {
private:
    ofstream shtooka_file;
    double audiotime = 0;
    AVCodecContext *audioOutputCodecContext = nullptr;
    AVStream *audioStream = nullptr;
    AVFormatContext *fc = nullptr;
    unsigned audframe = 0;
    vector<vector<float>> sample_buffer;
    vector<vector<float>>    sfx_buffer;
    int sample_buffer_offset = 0; // The index in the sample_buffer at which the latest macroblock starts, since some samples from the last one may not have flushed entirely.
    int total_samples_processed = 0;

    bool file_exists(const string& filename){
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }
    double encode_and_write_audio(){
        AVPacket outputPacket = {0};

        double length_in_seconds = 0;

        int ret = 1;
        while (ret >= 0) {
            ret = avcodec_receive_packet(audioOutputCodecContext, &outputPacket);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }

            // Set the stream index of the output packet to the audio stream index
            outputPacket.stream_index = audioStream->index;

            // Set the correct PTS and DTS values for the output packet
            outputPacket.dts = outputPacket.pts = av_rescale_q(audiotime, audioStream->time_base, audioOutputCodecContext->time_base);
            pts_dts_plot.add_datapoint(vector<double>{static_cast<double>(outputPacket.pts)});
            audiotime += outputPacket.duration;

            length_in_seconds += static_cast<double>(outputPacket.duration) / audioOutputCodecContext->sample_rate;

            // Rescale PTS and DTS values before writing the packet
            av_packet_rescale_ts(&outputPacket, audioOutputCodecContext->time_base, audioStream->time_base);

            ret = av_write_frame(fc, &outputPacket);

            av_packet_unref(&outputPacket);
        }
        return length_in_seconds;
    }

public:
    AudioWriter(AVFormatContext *fc_) : fc(fc_), sample_buffer(2), sfx_buffer(2) {
        shtooka_file.open(PATH_MANAGER.record_list_path);
        if (!shtooka_file.is_open()) {
            cerr << "Error opening recorder list: " << PATH_MANAGER.record_list_path << endl;
        }

        // Set up codec for output
        const AVCodec* audioOutputCodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!audioOutputCodec) {
            throw runtime_error("Error: Could not find audio encoder.");
        }

        audioOutputCodecContext = avcodec_alloc_context3(audioOutputCodec);
        if (!audioOutputCodecContext) {
            throw runtime_error("Error: Could not allocate codec context for encoder.");
        }

        // Configure codec context
        audioOutputCodecContext->bit_rate = 128000; // 128 kbps
        audioOutputCodecContext->sample_rate = 44100; // 44.1 kHz
        av_channel_layout_default(&audioOutputCodecContext->ch_layout, 2); // 2 for stereo
        audioOutputCodecContext->sample_fmt = AV_SAMPLE_FMT_FLTP; // Floating-point planar format
        audioOutputCodecContext->time_base = {1, audioOutputCodecContext->sample_rate}; // Time base of 1/sample_rate

        // AAC-only
        audioOutputCodecContext->profile = FF_PROFILE_AAC_LOW; // Low complexity AAC (LC-AAC)

        // Open codec
        if (avcodec_open2(audioOutputCodecContext, audioOutputCodec, nullptr) < 0) {
            avcodec_free_context(&audioOutputCodecContext);
            throw runtime_error("Error: Could not open encoder.");
        }

        // Create an audio stream in the output format context
        audioStream = avformat_new_stream(fc, audioOutputCodec);
        if (!audioStream) {
            avcodec_free_context(&audioOutputCodecContext);
            throw runtime_error("Error: Could not create audio stream in output format context.");
        }

        audioStream->time_base = audioOutputCodecContext->time_base;

        // Copy codec parameters from codec context to the audio stream
        if (avcodec_parameters_from_context(audioStream->codecpar, audioOutputCodecContext) < 0) {
            avcodec_free_context(&audioOutputCodecContext);
            throw runtime_error("Error: Could not initialize stream codec parameters.");
        }
    }

    void add_sfx(const vector<float>& left_buffer, const vector<float>& right_buffer, int t) {
        if (left_buffer.size() != right_buffer.size()) {
            throw runtime_error("SFX buffer lengths do not match. Left: " + to_string(left_buffer.size()) + ", right: " + to_string(right_buffer.size()));
        }

        int numSamples = left_buffer.size();
        int sample_copy_start = t - total_samples_processed + sample_buffer_offset;
        if(sample_copy_start < 0) sample_copy_start = 0;
        int sample_copy_end = sample_copy_start + numSamples;

        // Pointers to the input buffers for each channel
        const vector<float>* input_buffers[2] = {&left_buffer, &right_buffer};

        // Ensure sfx_buffer has enough capacity and add samples
        for (int ch = 0; ch < 2; ++ch) {
            // Resize if necessary to accommodate the samples
            if (sfx_buffer[ch].size() < sample_copy_end) {
                sfx_buffer[ch].resize(sample_copy_end, 0.0f); // Extend channel with silence
            }

            // Add the SFX samples to the buffer
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
        return static_cast<double>(num_samples) / audioOutputCodecContext->sample_rate;
    }

    void add_silence(double duration) {
        process_frame_from_buffer();
        int num_samples = static_cast<int>(duration * audioOutputCodecContext->sample_rate);

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

        // Find the audio stream and ensure it is FLAC
        AVStream* audioStream = nullptr;
        for (unsigned int i = 0; i < inputAudioFormatContext->nb_streams; ++i) {
            if (inputAudioFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStream = inputAudioFormatContext->streams[i];
                break;
            }
        }
        if (!audioStream) {
            throw runtime_error("Error: No audio stream found in file: " + fullInputAudioFilename);
        }

        if (audioStream->codecpar->codec_id != AV_CODEC_ID_AAC) {
            throw runtime_error("Error: Input file is not in expected format: " + fullInputAudioFilename);
        }

        // Check sample rate
        if (audioStream->codecpar->sample_rate != 44100) {
            throw runtime_error("Error: Unsupported sample rate: " + to_string(audioStream->codecpar->sample_rate) + ". Expected 44100 Hz.");
        }

        // Ensure the audio is stereo
        int num_channels = 0;
        if (audioStream->codecpar->ch_layout.order != AV_CHANNEL_ORDER_UNSPEC) {
            num_channels = audioStream->codecpar->ch_layout.nb_channels;
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

        if (avcodec_parameters_to_context(codecContext, audioStream->codecpar) < 0) {
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
            if (packet->stream_index == audioStream->index) {
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

                    // Map channels to sample buffer
                    for (int i = 0; i < num_samples; ++i) {
                        sample_buffer[0].push_back(reinterpret_cast<float*>(frame->data[0])[i]); // Left channel
                        sample_buffer[1].push_back(reinterpret_cast<float*>(frame->data[1])[i]); // Right channel
                    }

                    // Update total length
                    length_in_seconds += static_cast<double>(num_samples) / codecContext->sample_rate;

                    av_frame_unref(frame);
                }

                av_frame_free(&frame);
            }
            av_packet_unref(packet);
        }

        // Clean up
        avcodec_free_context(&codecContext);
        avformat_close_input(&inputAudioFormatContext);

        return length_in_seconds;
    }

    void process_frame_from_buffer() {
        while(true){
            int channels = audioOutputCodecContext->ch_layout.nb_channels;
            int frameSize = audioOutputCodecContext->frame_size;
            if(sample_buffer[0].size() != sample_buffer[1].size()) throw runtime_error("Audio planar buffer mismatch!");
            if(sample_buffer[0].size() < frameSize) break;
            for (int ch = 0; ch < channels; ++ch) {
                if (sfx_buffer[ch].size() < sample_buffer[ch].size()) {
                    sfx_buffer[ch].resize(sample_buffer[ch].size(), 0.0f); // Extend channel with silence
                }
            }

            // Create a frame and set its properties
            AVFrame* frame = av_frame_alloc();
            if (!frame) { throw runtime_error("Frame allocation failed."); }
            frame->nb_samples = frameSize;
            av_channel_layout_default(&frame->ch_layout, 2); // 2 for stereo
            frame->sample_rate = audioOutputCodecContext->sample_rate;
            frame->format = audioOutputCodecContext->sample_fmt;

            // Allocate buffer for the frame
            int ret = av_frame_get_buffer(frame, 0);
            if (ret < 0) {
                av_frame_free(&frame);
                throw runtime_error("Error allocating audio frame buffer.");
            }

            for (int ch = 0; ch < channels; ++ch) {
                float* dst = reinterpret_cast<float*>(frame->data[ch]);
                // Ensure bounds for `sample_buffer`, `sfx_buffer`, and `frame->data[ch]`
                if (sample_buffer[ch].size() < frameSize || sfx_buffer[ch].size() < frameSize) {
                    throw runtime_error("Audio buffer size is smaller than the required frame size.");
                }

                for (int i = 0; i < frameSize; ++i) {
                    float sample = sample_buffer[ch][i] + sfx_buffer[ch][i]; // Perform element-wise addition
                    if(isnan(sample) || isinf(sample)) {
                        throw runtime_error("Audio was either inf or nan. SampleBuffer: " + to_string(sample_buffer[ch][i]) + ", SFXBuffer: " + to_string(sfx_buffer[ch][i]));
                    }
                    dst[i] = sample;
                }
            }

            frame->pts = audframe;
            audframe++;

            // Send the frame to the encoder
            ret = avcodec_send_frame(audioOutputCodecContext, frame);

            if (ret < 0) {
                //av_frame_free(&frame);
                //throw runtime_error("Error sending frame to encoder.");
                // TODO This happens nominally. Wtf?
            }

            encode_and_write_audio();

            av_frame_unref(frame);
            av_frame_free(&frame);

            for (int ch = 0; ch < channels; ++ch) {
                sample_buffer[ch].erase(sample_buffer[ch].begin(), sample_buffer[ch].begin() + frameSize);
                   sfx_buffer[ch].erase(   sfx_buffer[ch].begin(),    sfx_buffer[ch].begin() + frameSize);
            }
            total_samples_processed += frameSize;
        }
        sample_buffer_offset = sample_buffer[0].size();
    }

    void add_shtooka_entry(const string& filename, const string& text) {
        if (!shtooka_file.is_open()) {
            std::cerr << "Shtooka file is not open. Cannot add entry." << std::endl;
            return;
        }

        shtooka_file << filename << "\t" << text << "\n";
    }
    void cleanup() {
        process_frame_from_buffer();
        avcodec_send_frame(audioOutputCodecContext, NULL);
        encode_and_write_audio();
        
        avcodec_free_context(&audioOutputCodecContext);
        
        if (shtooka_file.is_open()) {
            shtooka_file.close();
        }
    }
};
