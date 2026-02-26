#include "Macroblock.h"

#include <iostream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>
#include "../IO/Writer.h"

using namespace std;

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

SilenceBlock::SilenceBlock(const double duration_seconds)
    : duration_frames(duration_seconds * get_video_framerate_fps()) {
    if (duration_frames <= 0) {
        throw invalid_argument("Duration must be greater than 0");
    }
}

int SilenceBlock::write_and_get_duration_frames() const {
    get_writer().audio->add_silence(duration_frames);
    get_writer().subtitle->add_silence(static_cast<double>(duration_frames) / get_video_framerate_fps());
    return duration_frames;
}
string SilenceBlock::blurb() const { return "SilenceBlock(" + to_string(static_cast<double>(duration_frames) / get_video_framerate_fps()) + "s)"; }

FileBlock::FileBlock(const string& subtitle_text)
    : subtitle_text(subtitle_text), audio_filename(sanitize_filename(subtitle_text)) {}

void FileBlock::write_shtooka() const {
    get_writer().shtooka->add_shtooka_entry(audio_filename, subtitle_text);
}

int FileBlock::write_and_get_duration_frames() const {
    int duration_frames = get_writer().audio->add_audio_from_file(audio_filename);
    get_writer().subtitle->add_subtitle(static_cast<double>(duration_frames) / get_video_framerate_fps(), subtitle_text);
    return duration_frames;
}
string FileBlock::blurb() const { return "FileBlock(" + subtitle_text + ")"; }

GeneratedBlock::GeneratedBlock(const vector<int32_t>& leftBuffer, const vector<int32_t>& rightBuffer)
    : leftBuffer(leftBuffer), rightBuffer(rightBuffer) {
    if (leftBuffer.size() != rightBuffer.size()) {
        throw invalid_argument("Left and right buffers must have the same size");
    }
}

int GeneratedBlock::write_and_get_duration_frames() const {
    int duration_frames = get_writer().audio->add_generated_audio(leftBuffer, rightBuffer);
    get_writer().subtitle->add_subtitle(static_cast<double>(duration_frames) / get_video_framerate_fps(), "[Computer Generated Sound]");
    return duration_frames;
}
string GeneratedBlock::blurb() const { return "GeneratedBlock(" + to_string(leftBuffer.size() / get_audio_samplerate_hz()) + "s)"; }

CompositeBlock::CompositeBlock(const Macroblock& a, const Macroblock& b)
    : a(a), b(b) {}

void CompositeBlock::write_shtooka() const {
    a.write_shtooka();
    b.write_shtooka();
}

int CompositeBlock::write_and_get_duration_frames() const {
    int a_duration = a.write_and_get_duration_frames();
    int b_duration = b.write_and_get_duration_frames();
    return a_duration + b_duration;
}
string CompositeBlock::blurb() const { return "CompositeBlock(" + a.blurb() + ", " + b.blurb() + ")"; }
