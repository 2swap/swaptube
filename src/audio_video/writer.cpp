// Adapted from https://stackoverflow.com/questions/34511312

#pragma once

#include <filesystem>
#include "../misc/pixels.h"
extern "C"
{
    #include <libswscale/swscale.h>
    #include <libavformat/avformat.h>
}

char* err2str(int errnum)
{
    // static char str[AV_ERROR_MAX_STRING_SIZE];
    // thread_local may be better than static in multi-thread circumstance
    thread_local char str[AV_ERROR_MAX_STRING_SIZE]; 
    memset(str, 0, sizeof(str));
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}


using namespace std;

class MovieWriter
{
    unsigned outframe = 0, audframe = 0;
    int audiotime = 0;
    double substime = 0;
    int subtitle_count = 0;
    int audioStreamIndex;

    string output_filename, srt_filename, record_filename;
    string media_folder;

    ofstream srt_file;
    ofstream shtooka_file;

    SwsContext* sws_ctx = nullptr;
    AVStream* videoStream = nullptr;
    AVStream* audioStream = nullptr;
    AVFormatContext* fc = nullptr;
    AVCodecContext* videoCodecContext = nullptr;
    AVCodecContext* audioInputCodecContext = nullptr;
    AVCodecContext* audioOutputCodecContext = nullptr;
    AVPacket pkt = {0}, inputPacket = {0};

    AVFrame *rgbpic = nullptr;
    AVFrame *yuvpic = nullptr;

    ofstream audio_pts_file;

public:

    void make_media_folder() {
        // Check if the folder already exists
        if (filesystem::exists(media_folder)) {
            cout << "Media folder already exists, not creating." << endl;
        } else {
            // Create the folder
            if (filesystem::create_directory(media_folder)) {
                cout << "Media folder created successfully." << endl;
            } else {
                cout << "Failed to create media folder." << endl;
            }
        }
    }

    MovieWriter(const string& output_filename_, const string& srt_filename_, const string& record_filename_, const string& media_) :
    output_filename(output_filename_), srt_filename(srt_filename_), record_filename(record_filename_), inputPacket(), media_folder(media_) {}

    MovieWriter() {}

    bool file_exists(const std::string& filename);
    double add_audio_get_length(const string& inputAudioFilename);
    void add_silence(double duration);
    double encode_and_write_audio();
    bool encode_and_write_frame(AVFrame* frame);
    void addFrame(const Pixels& p);
    void set_audiotime(double t_seconds);
    void add_srt_time(double s);
    void add_subtitle(double duration, const string& text);
    double add_audio_segment(const AudioSegment& audio);
    void add_shtooka_entry(const string& filename, const string& subtitleText);
    void init_audio(const string& inputAudioFilename);
    void init_video();
    void destroy_audio();
    void destroy_video();

    void init(const string& inputAudioFilename) {
        make_media_folder();
        avformat_alloc_output_context2(&fc, NULL, NULL, output_filename.c_str());
        av_log_set_level(AV_LOG_DEBUG);

        init_audio(inputAudioFilename);
        init_video();
    }

    ~MovieWriter()
    {
        destroy_audio();

        destroy_video();
        
        cout << "closing the file" << endl;
        av_write_trailer(fc);
        avio_closep(&fc->pb);
        avformat_free_context(fc);
    }

};

#include "audio.cpp"
#include "video.cpp"
#include "subs.cpp"
