#pragma once

#include "../misc/pixels.h"
#include "subs.cpp"
#include "audio.cpp"
#include "video.cpp"

char* err2str(int errnum)
{
    thread_local char str[AV_ERROR_MAX_STRING_SIZE]; 
    memset(str, 0, sizeof(str));
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}

using namespace std;

double video_seconds_so_far = 0;
double audio_seconds_so_far = 0;

class MovieWriter
{
private:
    bool made_folder;
    AVFormatContext* fc = nullptr;
    SubtitleWriter subswriter;
    AudioWriter audiowriter;
    VideoWriter videowriter;

    // Function to initialize 'fc'
    AVFormatContext* initFC() {
        AVFormatContext* temp_fc = nullptr;
        avformat_alloc_output_context2(&temp_fc, NULL, NULL, PATH_MANAGER.video_output.c_str());
        return temp_fc;
    }

public:
    MovieWriter()
    : fc(initFC()),
      subswriter(),
      audiowriter(fc),
      videowriter(fc) {
        init();
    }

    ~MovieWriter(){
        // We babysit these cleanup functions instead of delegating to destructors because
        // the shared fc resource complicates things and ordering of cleanup is crucial.
        audiowriter.cleanup();
        videowriter.cleanup();
    }

    void init() {
        audiowriter.init_audio();
        videowriter.init_video();
    }

    void set_time(double t_seconds){
        subswriter.set_substime(t_seconds);
        audiowriter.set_audiotime(t_seconds);
    }

    void add_frame(const Pixels& p) {
        videowriter.add_frame(p);
        video_seconds_so_far += 1./VIDEO_FRAMERATE;
    }

    void add_shtooka(const AudioSegment& audio){
        if (!audio.is_silence())
            audiowriter.add_shtooka_entry(audio.get_audio_filename(), audio.get_subtitle_text());
    }

    double add_audio_segment(const AudioSegment& audio){
        double duration_seconds = 0;
        if (audio.is_silence()) {
            duration_seconds = audio.get_duration_seconds();
            audiowriter.add_silence(duration_seconds);
        } else {
            duration_seconds = audiowriter.add_audio_get_length(audio.get_audio_filename());
            subswriter.add_subtitle(duration_seconds, audio.get_subtitle_text());
        }
        audio_seconds_so_far += duration_seconds;
        return audio_seconds_so_far - video_seconds_so_far;
    }

};

// This is a global writer handle which is accessed both by the main class and the scene object.
MovieWriter WRITER;
