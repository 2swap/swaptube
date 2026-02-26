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
    virtual int write_and_get_duration_frames() const = 0;
};

class SilenceBlock : public Macroblock {
public:
    SilenceBlock(const double duration_seconds);

    int write_and_get_duration_frames() const override;
    string blurb() const override;

private:
    const int duration_frames;
};

class FileBlock : public Macroblock {
public:
    FileBlock(const string& subtitle_text);

    void write_shtooka() const override;
    int write_and_get_duration_frames() const override;
    string blurb() const override;

private:
    const string subtitle_text;
    const string audio_filename;
};

class GeneratedBlock : public Macroblock {
public:
    GeneratedBlock(const vector<int32_t>& leftBuffer, const vector<int32_t>& rightBuffer);

    int write_and_get_duration_frames() const override;
    string blurb() const override;

private:
    const vector<int32_t> leftBuffer;
    const vector<int32_t> rightBuffer;
};

class CompositeBlock : public Macroblock {
public:
    CompositeBlock(const Macroblock& a, const Macroblock& b);

    void write_shtooka() const override;
    int write_and_get_duration_frames() const override;
    string blurb() const override;

private:
    const Macroblock& a;
    const Macroblock& b;
};
