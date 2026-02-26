#pragma once

#include <fstream>
#include <string>

class SubtitleWriter {
private:
    std::ofstream srt_file;
    double substime = 0;
    int subtitle_count = 0;
    std::string last_written_subtitle = "";

    void add_srt_time(double s);

public:
    SubtitleWriter();
    ~SubtitleWriter();

    std::string get_last_written_subtitle() const;

    void get_substime(double t_seconds);
    void add_silence(double duration);
    void add_subtitle(double duration, const std::string& text);
};
