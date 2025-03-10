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
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/channel_layout.h>
    #include <libswscale/swscale.h>
    #include <libavformat/avformat.h>
}

using namespace std;


class AudioWriter {
private:
    ofstream /*audio_pts_file,*/ shtooka_file;
    // Persistent buffer for each channel – must be initialized once (e.g., in your class constructor)
std::vector<std::vector<float>> globalAudioBuffers;

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
            failout("Error: Input audio file " + PATH_MANAGER.testaudio_path + " does not exist.");
        }

        AVFormatContext* inputAudioFormatContext = nullptr;
        if (avformat_open_input(&inputAudioFormatContext, PATH_MANAGER.testaudio_path.c_str(), nullptr, nullptr) < 0) {
            failout("Error: Could not open input audio file " + PATH_MANAGER.testaudio_path);
        }

        // Find input audio stream information
        if (avformat_find_stream_info(inputAudioFormatContext, nullptr) < 0){
            avformat_close_input(&inputAudioFormatContext);
            failout("Error: Could not find stream information in input audio file " + PATH_MANAGER.testaudio_path);
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

    void set_audiotime(double t_seconds) {
        double t_samples = audioOutputCodecContext->sample_rate * t_seconds;
        /*if(t_samples < audiotime){
            cerr << "Audio PTS latchup!" << endl << "Was: " << audiotime << " and is being set to " << t_samples << "!" << endl << "Aborting!" << endl;
            exit(1);
        }*/
        audiotime = t_samples;
    }
void add_silence(double duration) {
    // Determine how many samples are needed.
    int numSamples = static_cast<int>(duration * audioOutputCodecContext->sample_rate);
    int frameSize  = audioOutputCodecContext->frame_size;
    int channels   = audioOutputCodecContext->channels;
    int sample_rate = audioOutputCodecContext->sample_rate;

    // Ensure the persistent buffer is set up.
    if (globalAudioBuffers.empty() || globalAudioBuffers.size() != (size_t)channels) {
        globalAudioBuffers.resize(channels);
    }

    // Append silence (0.0f) for each channel.
    for (int ch = 0; ch < channels; ch++) {
        globalAudioBuffers[ch].insert(globalAudioBuffers[ch].end(), numSamples, 0.0f);
    }

    // Flush complete frames from the persistent buffer.
    while (globalAudioBuffers[0].size() >= static_cast<size_t>(frameSize)) {
        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            failout("Could not allocate silence frame.");
        }
        frame->nb_samples    = frameSize;
        frame->channel_layout = audioOutputCodecContext->channel_layout;
        frame->format         = audioOutputCodecContext->sample_fmt; // fltp
        frame->sample_rate    = sample_rate;

        int ret = av_frame_get_buffer(frame, 0);
        if (ret < 0) {
            failout("Could not allocate silence frame samples.");
        }

        for (int ch = 0; ch < channels; ch++) {
            memcpy(frame->data[ch],
                   globalAudioBuffers[ch].data(),
                   frameSize * sizeof(float));
            // Remove the flushed samples.
            globalAudioBuffers[ch].erase(globalAudioBuffers[ch].begin(),
                                         globalAudioBuffers[ch].begin() + frameSize);
        }

        frame->pts = audframe;
        audframe++;
        ret = avcodec_send_frame(audioOutputCodecContext, frame);
        if (ret < 0) {
            failout("Error sending silence frame to encoder.");
        }
        encode_and_write_audio();

        av_frame_unref(frame);
        av_frame_free(&frame);
    }
}

double add_audio_get_length(const string& audioname) {
    double length_in_seconds = 0;
    string fullInputAudioFilename = PATH_MANAGER.this_project_media_dir + audioname;

    // If file doesn't exist, delegate to add_silence.
    if (!file_exists(fullInputAudioFilename)) {
        add_silence(3.0);
        return 3.0;
    }

    AVFormatContext* inputAudioFormatContext = nullptr;
    int ret = avformat_open_input(&inputAudioFormatContext, fullInputAudioFilename.c_str(), nullptr, nullptr);
    if (ret < 0) {
        failout("Error opening input audio file.");
    }

    // Get frame size, channels, and sample rate from output codec context.
    int frame_size  = audioOutputCodecContext->frame_size;
    int channels    = audioOutputCodecContext->channels;
    int sample_rate = audioOutputCodecContext->sample_rate;

    // Ensure the persistent buffer is set up.
    if (globalAudioBuffers.empty() || globalAudioBuffers.size() != (size_t)channels) {
        globalAudioBuffers.resize(channels);
    }

    AVPacket inputPacket;
    av_init_packet(&inputPacket);

    while (av_read_frame(inputAudioFormatContext, &inputPacket) >= 0) {
        if (inputPacket.stream_index == audioStreamIndex) {
            AVFrame* frame = av_frame_alloc();
            if (!frame) {
                failout("Could not allocate input frame.");
            }

            ret = avcodec_send_packet(audioInputCodecContext, &inputPacket);
            if (ret < 0) {
                failout("Error sending packet to decoder.");
            }

            // Process decoded frames.
            while ((ret = avcodec_receive_frame(audioInputCodecContext, frame)) >= 0) {
                int nb_samples = frame->nb_samples;
                // Append samples from each channel to the persistent buffer.
                for (int ch = 0; ch < channels; ch++) {
                    float* src = reinterpret_cast<float*>(frame->data[ch]);
                    globalAudioBuffers[ch].insert(globalAudioBuffers[ch].end(), src, src + nb_samples);
                }

                // Flush complete frames (frame_size samples per channel).
                while (globalAudioBuffers[0].size() >= static_cast<size_t>(frame_size)) {
                    AVFrame* outFrame = av_frame_alloc();
                    if (!outFrame) {
                        failout("Could not allocate output frame.");
                    }
                    outFrame->nb_samples    = frame_size;
                    outFrame->channel_layout = audioOutputCodecContext->channel_layout;
                    outFrame->format         = audioOutputCodecContext->sample_fmt; // fltp
                    outFrame->sample_rate    = sample_rate;

                    ret = av_frame_get_buffer(outFrame, 0);
                    if (ret < 0) {
                        failout("Could not allocate output frame samples.");
                    }

                    // Copy exactly frame_size samples from each channel.
                    for (int ch = 0; ch < channels; ch++) {
                        memcpy(outFrame->data[ch],
                               globalAudioBuffers[ch].data(),
                               frame_size * sizeof(float));
                        // Remove the samples that have been used.
                        globalAudioBuffers[ch].erase(globalAudioBuffers[ch].begin(),
                                                     globalAudioBuffers[ch].begin() + frame_size);
                    }

                    ret = avcodec_send_frame(audioOutputCodecContext, outFrame);
                    if (ret < 0) {
                        failout("Error sending frame to encoder.");
                    }
                    length_in_seconds += encode_and_write_audio();

                    av_frame_free(&outFrame);
                }
                av_frame_unref(frame);
            }
            av_frame_free(&frame);
        }
        av_packet_unref(&inputPacket);
    }

    // Do not flush an incomplete frame – leave it in the persistent buffer.
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
