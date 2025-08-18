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
    return sanitized + ".wav";
}

class Macroblock {
public:
    virtual ~Macroblock() = default;
    double invoke_get_macroblock_length_seconds() const {
        AUDIO_WRITER.audio_seconds_so_far += write_and_get_duration_seconds();
        return AUDIO_WRITER.audio_seconds_so_far - VIDEO_WRITER.video_seconds_so_far;
    }
    virtual void write_shtooka() const {}
    virtual string blurb() const = 0; // This is how the macroblock identifies itself in log outputs
    virtual double write_and_get_duration_seconds() const = 0;
};

class SilenceBlock : public Macroblock {
public:
    SilenceBlock(const double duration_seconds)
        : duration_seconds(duration_seconds) {
        if (duration_seconds <= 0) {
            throw invalid_argument("Duration must be greater than 0");
        }
    }

    double write_and_get_duration_seconds() const override {
        AUDIO_WRITER.add_silence(duration_seconds);
        return duration_seconds;
    }
    string blurb() const override { return "SilenceBlock(" + to_string(duration_seconds) + ")"; }

private:
    const double duration_seconds;
};

class FileBlock : public Macroblock {
public:
    FileBlock(const string& subtitle_text)
        : subtitle_text(subtitle_text), audio_filename(sanitize_filename(subtitle_text)) {}

    void write_shtooka() const override {
        SHTOOKA_WRITER.add_shtooka_entry(audio_filename, subtitle_text);
    }

    double write_and_get_duration_seconds() const override {
        double duration_seconds = AUDIO_WRITER.add_audio_from_file(audio_filename);
        SUBTITLE_WRITER.add_subtitle(duration_seconds, subtitle_text);
        return duration_seconds;
    }
    string blurb() const override { return "FileBlock(" + subtitle_text + ")"; }

private:
    const string subtitle_text;
    const string audio_filename;
};

class GeneratedBlock : public Macroblock {
public:
    GeneratedBlock(const vector<int32_t>& leftBuffer, const vector<int32_t>& rightBuffer)
        : leftBuffer(leftBuffer), rightBuffer(rightBuffer) {
        if (leftBuffer.size() != rightBuffer.size()) {
            throw invalid_argument("Left and right buffers must have the same size");
        }
    }

    double write_and_get_duration_seconds() const override {
        double duration_seconds = AUDIO_WRITER.add_generated_audio(leftBuffer, rightBuffer);
        SUBTITLE_WRITER.add_subtitle(duration_seconds, "[Computer Generated Sound]");
        return duration_seconds;
    }
    string blurb() const override { return "GeneratedBlock(" + to_string(leftBuffer.size()/SAMPLERATE) + ")"; }

private:
    const vector<int32_t> leftBuffer;
    const vector<int32_t> rightBuffer;
};

class CompositeBlock : public Macroblock {
public:
    CompositeBlock(const Macroblock& a, const Macroblock& b)
        : a(a), b(b) {}

    void write_shtooka() const override {
        a.write_shtooka();
        b.write_shtooka();
    }

    double write_and_get_duration_seconds() const override {
        return a.write_and_get_duration_seconds() + b.write_and_get_duration_seconds();
    }
    string blurb() const override { return "CompositeBlock(" + a.blurb() + ", " + b.blurb() + ")"; }

private:
    const Macroblock& a;
    const Macroblock& b;
};
