#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cctype>

using namespace std;

class SubtitleWriter {
private:
    ofstream srt_file;
    double substime = 0;
    int subtitle_count = 0;
    const string srt_filename;

    void add_srt_time(double s) {
        // Format the elapsed time and duration in HH:MM:SS,mmm format
        int hours = static_cast<int>(s / 3600);
        s -= hours * 3600;
        int minutes = static_cast<int>(s / 60);
        s -= minutes * 60;
        int seconds = static_cast<int>(s);
        s -= seconds;
        int milliseconds = static_cast<int>(s * 1000);

        srt_file << setfill('0')
                 << setw(2) << hours << ":"
                 << setw(2) << minutes << ":"
                 << setw(2) << seconds << ","
                 << setw(3) << milliseconds;
    }

public:
    SubtitleWriter() {
        srt_file.open(PATH_MANAGER.subtitle_output);
        if (!srt_file.is_open()) failout("Failed to open file: " + PATH_MANAGER.subtitle_output);
    }

    ~SubtitleWriter() {
        if (srt_file.is_open()) srt_file.close();
    }

    void set_substime(double t_seconds) {
        substime = t_seconds;
    }

    void add_subtitle(double duration, const string& text) {
        if (text.size() > 120) {
            cout << "Subtitle too long!" << endl;
            size_t center = text.size() / 2;

            // Find the nearest non-alphanumeric character to the center
            size_t left = center;
            size_t right = center;

            while (left > 0 || right < text.size() - 1) {
                if (left > 0 && !isalnum(text[left])) {
                    break;
                }
                if (right < text.size() - 1 && !isalnum(text[right])) {
                    break;
                }
                if (left > 0) left--;
                if (right < text.size() - 1) right++;
            }

            size_t split_pos = (left > 0 && !isalnum(text[left])) ? left : right;

            // Split the text into first and remaining parts
            string firstPart = text.substr(0, split_pos + 1);
            string remainingPart = text.substr(split_pos + 1);

            // Calculate modified duration for the first part
            double modifiedDuration = duration * (firstPart.size() / static_cast<double>(text.size()));

            // Write the first part of the subtitle
            add_subtitle(modifiedDuration, firstPart);

            // Call the function recursively for the remaining part
            add_subtitle(duration - modifiedDuration, remainingPart);
            return;
        }

        // Recursive base case
        subtitle_count++;
        //cout << "Adding subtitle: " << text << endl;

        // Write the complete subtitle entry to the file
        srt_file << subtitle_count << endl;
        add_srt_time(substime);
        srt_file << " --> ";
        substime += duration - 0.05;
        add_srt_time(substime);
        srt_file << endl << text << endl << endl;
    }
};

