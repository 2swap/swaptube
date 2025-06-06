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

// TODO should this class just be renamed to Macroblock?
class AudioSegment {
public:
    virtual ~AudioSegment() = default;
    double invoke_get_macroblock_length_seconds() const {
        AUDIO_WRITER.audio_seconds_so_far += write_and_get_duration_seconds();
        return AUDIO_WRITER.audio_seconds_so_far - VIDEO_WRITER.video_seconds_so_far;
    }
    virtual void write_shtooka() const {}
private:
    virtual double write_and_get_duration_seconds() const = 0;
};

class SilenceSegment : public AudioSegment {
public:
    SilenceSegment(const double duration_seconds)
        : duration_seconds(duration_seconds) {
        if (duration_seconds <= 0) {
            throw invalid_argument("Duration must be greater than 0");
        }
    }

private:
    double write_and_get_duration_seconds() const override {
        AUDIO_WRITER.add_silence(duration_seconds);
        return duration_seconds;
    }

    const double duration_seconds;
};

class FileSegment : public AudioSegment {
public:
    FileSegment(const string& subtitle_text)
        : subtitle_text(subtitle_text), audio_filename(sanitize_filename(subtitle_text)) {}

    void write_shtooka() const override {
        SHTOOKA_WRITER.add_shtooka_entry(audio_filename, subtitle_text);
    }

private:
    double write_and_get_duration_seconds() const override {
        double duration_seconds = AUDIO_WRITER.add_audio_from_file(audio_filename);
        SUBTITLE_WRITER.add_subtitle(duration_seconds, subtitle_text);
        return duration_seconds;
    }

    const string subtitle_text;
    const string audio_filename;
};

class GeneratedSegment : public AudioSegment {
public:
    GeneratedSegment(const vector<float>& leftBuffer, const vector<float>& rightBuffer)
        : leftBuffer(leftBuffer), rightBuffer(rightBuffer) {
        if (leftBuffer.size() != rightBuffer.size()) {
            throw invalid_argument("Left and right buffers must have the same size");
        }
    }

private:
    double write_and_get_duration_seconds() const override {
        double duration_seconds = AUDIO_WRITER.add_generated_audio(leftBuffer, rightBuffer);
        SUBTITLE_WRITER.add_subtitle(duration_seconds, "[Computer Generated Sound]");
        return duration_seconds;
    }

    const vector<float> leftBuffer;
    const vector<float> rightBuffer;
};

class CompositeSegment : public AudioSegment {
public:
    CompositeSegment(const AudioSegment a, const AudioSegment b) {
        if (leftBuffer.size() != rightBuffer.size()) {
            throw invalid_argument("Left and right buffers must have the same size");
        }
    }

    void write_shtooka() const override {
        a.write_shtooka();
        b.write_shtooka();
    }

private:
    double write_and_get_duration_seconds() const override {
        return a.write_and_get_duration_seconds() + b.write_and_get_duration_seconds();
    }
};
