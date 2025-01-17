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
}

using namespace std;


class AudioWriter {
private:
    ofstream shtooka_file;
    int audioStreamIndex;
    double audiotime = 0;
    double substime = 0;
    AVCodecContext *audioInputCodecContext = nullptr;
    AVCodecContext *audioOutputCodecContext = nullptr;
    AVStream *audioStream = nullptr;
    AVFormatContext *fc = nullptr;
    AVPacket inputPacket = {0};
    unsigned audframe = 0;
    vector<vector<float>> sampleBuffer;
    int sampleBuffer_offset = 0;

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
    AudioWriter(AVFormatContext *fc_) : fc(fc_), sampleBuffer(2) {
        shtooka_file.open(PATH_MANAGER.record_list_path);
        if (!shtooka_file.is_open()) {
            cerr << "Error opening recorder list: " << PATH_MANAGER.record_list_path << endl;
        }
    }

    void init_audio() {
        // Check if the input file exists
        if (!filesystem::exists(PATH_MANAGER.testaudio_path)) {
            throw runtime_error("Error: Input audio file " + PATH_MANAGER.testaudio_path + " does not exist.");
        }

        AVFormatContext* inputAudioFormatContext = nullptr;
        if (avformat_open_input(&inputAudioFormatContext, PATH_MANAGER.testaudio_path.c_str(), nullptr, nullptr) < 0) {
            throw runtime_error("Error: Could not open input audio file " + PATH_MANAGER.testaudio_path);
        }

        // Find input audio stream information
        if (avformat_find_stream_info(inputAudioFormatContext, nullptr) < 0){
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Could not find stream information in input audio file " + PATH_MANAGER.testaudio_path);
        }

        // Find input audio stream
        audioStreamIndex = -1;
        AVCodecParameters* codecParams = nullptr;
        for (unsigned int i = 0; i < inputAudioFormatContext->nb_streams; ++i) {
            if (inputAudioFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndex = i;
                codecParams = inputAudioFormatContext->streams[i]->codecpar;
                break;
            }
        }

        // INPUT
        const AVCodec* audioInputCodec = avcodec_find_decoder(codecParams->codec_id);
        audioInputCodecContext = avcodec_alloc_context3(audioInputCodec);
        avcodec_parameters_to_context(audioInputCodecContext, codecParams);
        avcodec_open2(audioInputCodecContext, audioInputCodec, nullptr);

        // Create a new audio stream in the output format context
        // Copy codec parameters to the new audio stream
        audioStream = avformat_new_stream(fc, audioInputCodec);
        avcodec_parameters_copy(audioStream->codecpar, codecParams);

        // OUTPUT
        const AVCodec* audioOutputCodec = avcodec_find_encoder(codecParams->codec_id);
        audioOutputCodecContext = avcodec_alloc_context3(audioOutputCodec);
        avcodec_parameters_to_context(audioOutputCodecContext, codecParams);
        avcodec_open2(audioOutputCodecContext, audioOutputCodec, nullptr);

        avformat_close_input(&inputAudioFormatContext);
    }

    double add_generated_audio(const vector<float>& leftBuffer, const vector<float>& rightBuffer) {
        process_frame_from_buffer();
        if (leftBuffer.size() != rightBuffer.size()) {
            throw runtime_error("Generated sound buffer lengths do not match. Left: "+ to_string(leftBuffer.size()) + ", right: " + to_string(rightBuffer.size()));
        }

        int numSamples = leftBuffer.size();

        for(int i = 0; i < numSamples; i++){
            sampleBuffer[0].push_back( leftBuffer[i]);
            sampleBuffer[1].push_back(rightBuffer[i]);
        }
        return static_cast<double>(numSamples) / audioOutputCodecContext->sample_rate;
    }

    void add_silence(double duration) {
        process_frame_from_buffer();
        int numSamples = static_cast<int>(duration * audioOutputCodecContext->sample_rate);

        for (int i = 0; i < numSamples; ++i) {
            sampleBuffer[0].push_back(0);
            sampleBuffer[1].push_back(0);
        }
    }

    double add_audio_from_file(const string& filename) {
        double length_in_seconds = 0;

        // Build full path to the input audio file
        string fullInputAudioFilename = PATH_MANAGER.this_project_media_dir + filename;

        // Check if the file exists
        if (!file_exists(fullInputAudioFilename)) {
            throw runtime_error("Error: Audio file not found: " + fullInputAudioFilename);
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
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: No audio stream found in file: " + fullInputAudioFilename);
        }

        if (audioStream->codecpar->codec_id != AV_CODEC_ID_FLAC) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Input file is not in FLAC format: " + fullInputAudioFilename);
        }

        // Check sample rate
        if (audioStream->codecpar->sample_rate != 44100) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Unsupported sample rate: " + to_string(audioStream->codecpar->sample_rate) + ". Expected 44100 Hz.");
        }

        // Ensure the audio is stereo
        if (audioStream->codecpar->ch_layout.nb_channels != 2) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Unsupported channel count: " + to_string(audioStream->codecpar->ch_layout.nb_channels) + ". Expected stereo (2 channels).");
        }

        // Set up the FLAC decoder
        const AVCodec* flacDecoder = avcodec_find_decoder(AV_CODEC_ID_FLAC);
        if (!flacDecoder) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: FLAC decoder not found.");
        }

        AVCodecContext* codecContext = avcodec_alloc_context3(flacDecoder);
        if (!codecContext) {
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Could not allocate codec context for FLAC decoder.");
        }

        if (avcodec_parameters_to_context(codecContext, audioStream->codecpar) < 0) {
            avcodec_free_context(&codecContext);
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Could not initialize codec context from stream parameters.");
        }

        if (avcodec_open2(codecContext, flacDecoder, nullptr) < 0) {
            avcodec_free_context(&codecContext);
            avformat_close_input(&inputAudioFormatContext);
            throw runtime_error("Error: Could not open FLAC decoder.");
        }

        // Decode and process the audio samples
        AVPacket packet;
        av_init_packet(&packet);
        while (av_read_frame(inputAudioFormatContext, &packet) >= 0) {
            if (packet.stream_index == audioStream->index) {
                AVFrame* frame = av_frame_alloc();
                if (!frame) {
                    av_packet_unref(&packet);
                    avcodec_free_context(&codecContext);
                    avformat_close_input(&inputAudioFormatContext);
                    throw runtime_error("Error: Could not allocate frame.");
                }

                int ret = avcodec_send_packet(codecContext, &packet);
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codecContext, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    }

                    int numSamples = frame->nb_samples;

                    // Map channels to sample buffer
                    for (int i = 0; i < numSamples; ++i) {
                        sampleBuffer[0].push_back(reinterpret_cast<float*>(frame->data[0])[i]); // Left channel
                        sampleBuffer[1].push_back(reinterpret_cast<float*>(frame->data[1])[i]); // Right channel
                    }

                    // Update total length
                    length_in_seconds += static_cast<double>(numSamples) / codecContext->sample_rate;

                    av_frame_unref(frame);
                }

                av_frame_free(&frame);
            }
            av_packet_unref(&packet);
        }

        // Clean up
        avcodec_free_context(&codecContext);
        avformat_close_input(&inputAudioFormatContext);

        cout << "Audio added successfully, length " << length_in_seconds << " seconds." << endl;

        return length_in_seconds;
    }

    void process_frame_from_buffer() {
        while(true){
            int channels = audioOutputCodecContext->ch_layout.nb_channels;
            int frameSize = audioOutputCodecContext->frame_size;
            if(sampleBuffer[0].size() != sampleBuffer[1].size()) throw runtime_error("Audio planar buffer mismatch!");
            if(sampleBuffer[0].size() < frameSize) break;

            // Create a frame and set its properties
            AVFrame* frame = av_frame_alloc();
            if (!frame) { throw runtime_error("Frame allocation failed."); }
            frame->nb_samples = frameSize;
            frame->ch_layout = audioOutputCodecContext->ch_layout;
            frame->sample_rate = audioOutputCodecContext->sample_rate;
            frame->format = audioOutputCodecContext->sample_fmt;

            // Allocate buffer for the frame
            int ret = av_frame_get_buffer(frame, 0);
            if (ret < 0) {
                av_frame_free(&frame);
                throw runtime_error("Error allocating audio frame buffer.");
            }

            for (int ch = 0; ch < channels; ++ch) {
                memcpy(frame->data[ch], 
                       sampleBuffer[ch].data(), 
                       frameSize * sizeof(float));
            }

            frame->pts = audframe;
            audframe++;

            // Send the frame to the encoder
            ret = avcodec_send_frame(audioOutputCodecContext, frame);

            if (ret < 0) {
                //av_frame_free(&frame);
                //throw runtime_error("Error sending frame to encoder.");
                // This happens nominally. Wtf?
            }

            encode_and_write_audio();

            av_frame_unref(frame);
            av_frame_free(&frame);

            for (int ch = 0; ch < channels; ++ch) {
                sampleBuffer[ch].erase(sampleBuffer[ch].begin(), sampleBuffer[ch].begin() + frameSize);
            }
        }
        sampleBuffer_offset = sampleBuffer[0].size();
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
        av_packet_unref(&inputPacket);
        
        avcodec_free_context(&audioOutputCodecContext);
        avcodec_free_context(&audioInputCodecContext);
        
        if (shtooka_file.is_open()) {
            shtooka_file.close();
        }
    }
};
