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

class MovieWriter
{
private:
    string media_folder, output_filename;
    AVFormatContext* fc = nullptr;
    SubtitleWriter subswriter;
    AudioWriter audiowriter;
    VideoWriter videowriter;

    // Function to initialize 'fc'
    AVFormatContext* initFC(const string& output_filename) {
        AVFormatContext* temp_fc = nullptr;
        avformat_alloc_output_context2(&temp_fc, NULL, NULL, output_filename.c_str());
        return temp_fc;
    }

public:

    MovieWriter(const string& project_name):
        media_folder("../media/" + project_name + "/"),
        output_filename("../out/" + project_name + ".mp4"),
        fc(initFC(output_filename)),
        subswriter(project_name),
        audiowriter(media_folder, fc),
        videowriter(output_filename, fc)
    {
        // Ensure necessary directories exist
        ensure_directory_exists("../media/");
        ensure_directory_exists(media_folder);
        ensure_directory_exists("../out/");
    }

    ~MovieWriter(){
        // We babysit these cleanup functions instead of delegating to destructors because
        // the shared fc resource complicates things and ordering of cleanup is crucial.
        audiowriter.cleanup();
        videowriter.cleanup();

        av_write_trailer(fc);
        avio_closep(&fc->pb);
        avformat_free_context(fc);
    }

    void init(const string& inputAudioFilename) {
        audiowriter.init_audio(inputAudioFilename);
        videowriter.init_video();
    }

    void set_time(double t_seconds){
        subswriter.set_substime(t_seconds);
        audiowriter.set_audiotime(t_seconds);
    }

    void add_frame(const Pixels& p) {
        videowriter.add_frame(p);
    }

    double add_audio_segment(const AudioSegment& audio){
        double duration_seconds = 0;
        if (audio.is_silence()) {
            duration_seconds = audio.get_duration_seconds();
            audiowriter.add_silence(duration_seconds);
        } else {
            duration_seconds = audiowriter.add_audio_get_length(audio.get_audio_filename());
            subswriter.add_subtitle(duration_seconds, audio.get_subtitle_text());
            audiowriter.add_shtooka_entry(audio.get_audio_filename(), audio.get_subtitle_text());
        }
        return duration_seconds;
    }

};

// This is a global writer handle which is accessed both by the main class and the scene object.
MovieWriter* WRITER;
