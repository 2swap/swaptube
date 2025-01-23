#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>

// Function to sanitize a string for use as a filename
string sanitize_filename(const string& text) {
    string sanitized = text;
    // Replace spaces with underscores
    replace(sanitized.begin(), sanitized.end(), ' ', '_');
    // Remove non-alphanumeric characters except underscores
    sanitized.erase(remove_if(sanitized.begin(), sanitized.end(),
        [](char c) { return !isalnum(c) && c != '_'; }),
        sanitized.end());
    return sanitized + ".aac";
}

class AudioSegment {
public:
    virtual ~AudioSegment() = default;
};

class SilenceSegment : public AudioSegment {
public:
    SilenceSegment(double duration_seconds)
        : duration_seconds(duration_seconds) {
        if (duration_seconds <= 0) {
            throw invalid_argument("Duration must be greater than 0");
        }
    }

    double get_duration_seconds() const {
        return duration_seconds;
    }

private:
    double duration_seconds;
};

class FileSegment : public AudioSegment {
public:
    FileSegment(const string& subtitle_text)
        : subtitle_text(subtitle_text), audio_filename(sanitize_filename(subtitle_text)) {}

    string get_audio_filename() const {
        return audio_filename;
    }

    string get_subtitle_text() const {
        return subtitle_text;
    }

private:
    string subtitle_text;
    string audio_filename;
};

class GeneratedSegment : public AudioSegment {
public:
    GeneratedSegment(const vector<float>& leftBuffer, const vector<float>& rightBuffer)
        : leftBuffer(leftBuffer), rightBuffer(rightBuffer) {
        if (leftBuffer.size() != rightBuffer.size()) {
            throw invalid_argument("Left and right buffers must have the same size");
        }
    }

    const vector<float>& get_left_buffer() const {
        return leftBuffer;
    }

    const vector<float>& get_right_buffer() const {
        return rightBuffer;
    }

private:
    vector<float> leftBuffer;
    vector<float> rightBuffer;
};
