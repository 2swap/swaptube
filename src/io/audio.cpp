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
extern "C"
{
    #include <libavformat/avformat.h>
}

using namespace std;


class AudioWriter {
private:
    ofstream /*audio_pts_file,*/ shtooka_file;
    int audioStreamIndex;
    double audiotime = 0;
    double substime = 0;
    AVCodecContext *audioInputCodecContext = nullptr;
    AVCodecContext *audioOutputCodecContext = nullptr;
    AVStream *audioStream = nullptr;
    AVFormatContext *fc = nullptr;
    AVPacket inputPacket = {0};
    unsigned audframe = 0;

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
            outputPacket.dts = av_rescale_q(audiotime, audioStream->time_base, audioOutputCodecContext->time_base);
            outputPacket.pts = outputPacket.dts;
            //audio_pts_file << outputPacket.pts << endl;
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
    AudioWriter(AVFormatContext *fc_) : fc(fc_) {
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
        AVCodec* audioInputCodec = avcodec_find_decoder(codecParams->codec_id);
        audioInputCodecContext = avcodec_alloc_context3(audioInputCodec);
        avcodec_parameters_to_context(audioInputCodecContext, codecParams);
        avcodec_open2(audioInputCodecContext, audioInputCodec, nullptr);

        // Create a new audio stream in the output format context
        // Copy codec parameters to the new audio stream
        audioStream = avformat_new_stream(fc, audioInputCodec);
        avcodec_parameters_copy(audioStream->codecpar, codecParams);

        // OUTPUT
        AVCodec* audioOutputCodec = avcodec_find_encoder(codecParams->codec_id);
        audioOutputCodecContext = avcodec_alloc_context3(audioOutputCodec);
        avcodec_parameters_to_context(audioOutputCodecContext, codecParams);
        avcodec_open2(audioOutputCodecContext, audioOutputCodec, nullptr);

        avformat_close_input(&inputAudioFormatContext);
    }

    void set_audiotime(double t_seconds) {
        double t_samples = audioOutputCodecContext->sample_rate * t_seconds;
        /*if(t_samples < audiotime){
            cerr << "Audio PTS latchup!" << endl << "Was: " << audiotime << " and is being set to " << t_samples << "!" << endl << "Aborting!" << endl;
            exit(1);
        }*/
        audiotime = t_samples;
    }
    void add_silence(double duration) {
        //cout << "Adding silence: " << duration << " seconds" << endl;

        // Calculate the number of samples needed for the specified duration
        int numSamples = static_cast<int>(duration * audioOutputCodecContext->sample_rate);

        // Calculate the frame size based on the codec context's frame size
        int frameSize = audioOutputCodecContext->frame_size;

        // Calculate the number of frames needed to accommodate the specified duration
        int numFrames = ceil(static_cast<double>(numSamples) / frameSize);

        // Allocate buffer for audio data
        int bufferSize = numSamples * audioOutputCodecContext->channels;
        vector<int16_t> audioBuffer(bufferSize, 0);

        // Split the audio data into multiple frames
        int samplesRemaining = numSamples;

        for (int i = 0; i < numFrames; i++) {
            int samples_this_frame = min(frameSize, samplesRemaining);

            // Fill the audio buffer with silence
            for (int ch = 0; ch < audioOutputCodecContext->channels; ch++) {
                int offset = ch * samples_this_frame;
                for (int s = 0; s < samples_this_frame; s++) {
                    audioBuffer[offset + s] = s%5;
                }
            }

            // Create a frame and set its properties
            AVFrame* frame = av_frame_alloc();
            frame->nb_samples = samples_this_frame;
            frame->channel_layout = audioOutputCodecContext->channel_layout;
            frame->sample_rate = audioOutputCodecContext->sample_rate;
            frame->format = audioOutputCodecContext->sample_fmt;

            // Fill the frame with the audio data
            int ret = av_frame_get_buffer(frame, 0);

            for (int ch = 0; ch < audioOutputCodecContext->channels; ch++) {
                frame->linesize[ch] = samples_this_frame * av_get_bytes_per_sample(audioOutputCodecContext->sample_fmt);
            }

            ret = avcodec_fill_audio_frame(frame, audioOutputCodecContext->channels, audioOutputCodecContext->sample_fmt,
                                           reinterpret_cast<const uint8_t*>(audioBuffer.data()), bufferSize, 0);

            frame->pts = audframe;
            audframe++;
            ret = avcodec_send_frame(audioOutputCodecContext, frame);

            encode_and_write_audio();

            av_frame_unref(frame);
            av_frame_free(&frame);

            samplesRemaining -= samples_this_frame;
        }
        //cout << "Added silence: " << duration << " seconds" << endl;
    }
    double add_audio_get_length(const string& audioname) {
        double length_in_seconds = 0;

        string fullInputAudioFilename = PATH_MANAGER.this_project_media_dir + audioname;

        // Check if the input audio file exists
        if (!file_exists(fullInputAudioFilename)) {
            length_in_seconds = 3.0;
            add_silence(length_in_seconds);
            return length_in_seconds;
        }

        AVFormatContext* inputAudioFormatContext = nullptr;
        int ret = avformat_open_input(&inputAudioFormatContext, fullInputAudioFilename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            throw runtime_error("Error opening input audio file.");
        }

        // Read input audio frames and write to output format context
        while (av_read_frame(inputAudioFormatContext, &inputPacket) >= 0) {
            if (inputPacket.stream_index == audioStreamIndex) {
                AVFrame* frame = av_frame_alloc();
                int ret = avcodec_send_packet(audioInputCodecContext, &inputPacket);

                while (ret >= 0) {
                    ret = avcodec_receive_frame(audioInputCodecContext, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    }

                    frame->pts = audframe;
                    audframe++;
                    avcodec_send_frame(audioOutputCodecContext, frame);

                    length_in_seconds += encode_and_write_audio();
                }
                av_frame_unref(frame);
                av_frame_free(&frame);
            }
        }

        avformat_close_input(&inputAudioFormatContext);

        cout << "Audio added successfully, length " << length_in_seconds << endl;

        return length_in_seconds;
    }
    void add_shtooka_entry(const string& filename, const string& text) {
        if (!shtooka_file.is_open()) {
            std::cerr << "Shtooka file is not open. Cannot add entry." << std::endl;
            return;
        }

        shtooka_file << filename << "\t" << text << "\n";
    }
    void cleanup() {
        avcodec_send_frame(audioOutputCodecContext, NULL);
        encode_and_write_audio();
        
        avcodec_free_context(&audioOutputCodecContext);
        avcodec_free_context(&audioInputCodecContext);
        
        av_packet_unref(&inputPacket);
        
        //audio_pts_file.close();
        if (shtooka_file.is_open()) {
            shtooka_file.close();
        }
    }
};
