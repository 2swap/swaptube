#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>

// Function to sanitize a string for use as a filename
std::string sanitize_filename(const std::string& text) {
    std::string sanitized = text;
    // Replace spaces with underscores
    std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
    // Remove non-alphanumeric characters except underscores
    sanitized.erase(std::remove_if(sanitized.begin(), sanitized.end(),
        [](char c) { return !std::isalnum(c) && c != '_'; }),
        sanitized.end());
    return sanitized + ".mp3";
}

class AudioSegment {
public:
    // Constructors
    AudioSegment(double duration_seconds) 
        : duration_seconds(duration_seconds), audio_filename(""), subtitle_text("") {}

    AudioSegment(const string& subtitle_text)
        : duration_seconds(0), audio_filename(sanitize_filename(subtitle_text)), subtitle_text(subtitle_text) {}

    AudioSegment(const string& subtitle_text, const string& filename)
        : duration_seconds(0), audio_filename(filename), subtitle_text(subtitle_text) {}

    // Function to check if the audio segment represents silence
    bool is_silence() const {
        return audio_filename.empty() && subtitle_text.empty() && duration_seconds > 0;
    }

    // Getter methods
    double get_duration_seconds() const {
        return duration_seconds;
    }

    std::string get_audio_filename() const {
        return audio_filename;
    }

    std::string get_subtitle_text() const {
        return subtitle_text;
    }

    void display() const {
        std::cout << "Duration (seconds): " << duration_seconds << std::endl;
        std::cout << "Audio Filename: " << audio_filename << std::endl;
        std::cout << "Subtitle Text: " << subtitle_text << std::endl;
    }

private:
    double duration_seconds;
    std::string audio_filename;
    std::string subtitle_text;
};
