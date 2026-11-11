#pragma once
using namespace std;

#include <string>
#include <vector>
#include <cstdint>

string sanitize_filename(const string& text);

class Macroblock {
public:
    virtual ~Macroblock() = default;
    virtual void write_shtooka() const {}
    virtual string blurb() const = 0; // This is how the macroblock identifies itself in log outputs
    // Writes this macroblock's audio and subtitles, and returns its exact length in audio samples.
    // Audio is the source of truth for timing: it is never padded to video frame boundaries,
    // so the audio track comes out identical at any framerate.
    virtual int64_t write_and_get_duration_samples() const = 0;
};

class SilenceBlock : public Macroblock {
public:
    SilenceBlock(const double duration_seconds);

    int64_t write_and_get_duration_samples() const override;
    string blurb() const override;

private:
    const int64_t duration_samples;
};

class FileBlock : public Macroblock {
public:
    FileBlock(const string& subtitle_text);

    void write_shtooka() const override;
    int64_t write_and_get_duration_samples() const override;
    string blurb() const override;

private:
    const string subtitle_text;
    const string audio_filename;
};

class GeneratedBlock : public Macroblock {
public:
    GeneratedBlock(const vector<int32_t>& leftBuffer, const vector<int32_t>& rightBuffer);

    int64_t write_and_get_duration_samples() const override;
    string blurb() const override;

private:
    const vector<int32_t> leftBuffer;
    const vector<int32_t> rightBuffer;
};

class CompositeBlock : public Macroblock {
public:
    CompositeBlock(const Macroblock& a, const Macroblock& b);

    void write_shtooka() const override;
    int64_t write_and_get_duration_samples() const override;
    string blurb() const override;

private:
    const Macroblock& a;
    const Macroblock& b;
};
