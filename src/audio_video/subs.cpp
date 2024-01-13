#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

class SubtitleWriter {
private:
    ofstream srt_file;
    double substime = 0;
    int subtitle_count = 0;
    string srt_filename;

    void add_srt_time(double s) {
        // Format the elapsed time and duration in HH:MM:SS,mmm format
        int hours = static_cast<int>(s / 3600);
        s -= hours*3600;
        int minutes = static_cast<int>(s / 60);
        s -= minutes*60;
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
    SubtitleWriter(const string& project_name) : srt_filename("../out/" + project_name + ".srt") {
        cout << "Constructing a SubtitleWriter" << endl;
        srt_file.open(srt_filename);
        if (!srt_file.is_open()) {
            cerr << "Failed to open file: " << srt_filename << endl;
        }
        cout << "Constructed a SubtitleWriter" << endl;
    }

    ~SubtitleWriter() {
        if (srt_file.is_open()) {
            srt_file.close();
        }
    }
    void set_substime(double t_seconds) {
        substime = t_seconds;
    }

    void add_subtitle(double duration, const string& text) {
        if (text.size() > 90) {
            cout << "Subtitle too long!" << endl;
            // Find the position of the first period in the substring
            for (char c : {'!', '?', '.', ',', '-'}) {
                cout << "Attempting to split by " << c << endl;
                size_t firstPeriodPos = text.find(c);

                if (firstPeriodPos != string::npos) {
                    // Split the text into first and remaining parts
                    string firstPart = text.substr(0, firstPeriodPos + 1);
                    string remainingPart = text.substr(firstPeriodPos + 1);
                    if(remainingPart.size() < 12 || firstPart.size() < 12) continue;

                    // Calculate modified duration for the first part
                    double modifiedDuration = duration * (firstPart.size() / static_cast<double>(text.size()));

                    // Write the first part of the subtitle
                    add_subtitle(modifiedDuration, firstPart);

                    // Call the function recursively for the remaining part
                    add_subtitle(duration - modifiedDuration, remainingPart);
                    return;
                }
            }
        }
        //recursive base case
        subtitle_count++;
        cout << "Adding subtitle: " << text << endl;
        // Write the complete subtitle entry to the file
        srt_file << subtitle_count << endl;
        add_srt_time(substime);
        srt_file << " --> ";
        substime += duration - 0.05;
        add_srt_time(substime);
        srt_file << endl << text << endl << endl;
    }
};
