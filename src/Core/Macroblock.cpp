#include "Macroblock.h"

#include <iostream>
#include <algorithm>
#include <cmath>
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

static double samples_to_seconds(const int64_t samples) {
    return static_cast<double>(samples) / get_audio_samplerate_hz();
}

SilenceBlock::SilenceBlock(const double duration_seconds)
    : duration_samples(llround(duration_seconds * get_audio_samplerate_hz())) {
    if (duration_samples <= 0) {
        throw invalid_argument("Duration must be greater than 0");
    }
}

int64_t SilenceBlock::write_and_get_duration_samples() const {
    const int64_t samples = get_writer().audio->add_silence(duration_samples);
    get_writer().subtitle->add_silence(samples_to_seconds(samples));
    return samples;
}
string SilenceBlock::blurb() const { return "SilenceBlock(" + to_string(samples_to_seconds(duration_samples)) + "s)"; }

FileBlock::FileBlock(const string& subtitle_text)
    : subtitle_text(subtitle_text), audio_filename(sanitize_filename(subtitle_text)) {}

void FileBlock::write_shtooka() const {
    get_writer().shtooka->add_shtooka_entry(audio_filename, subtitle_text);
}

int64_t FileBlock::write_and_get_duration_samples() const {
    const int64_t samples = get_writer().audio->add_audio_from_file(audio_filename);
    get_writer().subtitle->add_subtitle(samples_to_seconds(samples), subtitle_text);
    return samples;
}
string FileBlock::blurb() const { return "FileBlock(" + subtitle_text + ")"; }

GeneratedBlock::GeneratedBlock(const vector<int32_t>& leftBuffer, const vector<int32_t>& rightBuffer)
    : leftBuffer(leftBuffer), rightBuffer(rightBuffer) {
    if (leftBuffer.size() != rightBuffer.size()) {
        throw invalid_argument("Left and right buffers must have the same size");
    }
}

int64_t GeneratedBlock::write_and_get_duration_samples() const {
    const int64_t samples = get_writer().audio->add_generated_audio(leftBuffer, rightBuffer);
    get_writer().subtitle->add_subtitle(samples_to_seconds(samples), "[Computer Generated Sound]");
    return samples;
}
string GeneratedBlock::blurb() const { return "GeneratedBlock(" + to_string(leftBuffer.size() / get_audio_samplerate_hz()) + "s)"; }

CompositeBlock::CompositeBlock(const Macroblock& a, const Macroblock& b)
    : a(a), b(b) {}

void CompositeBlock::write_shtooka() const {
    a.write_shtooka();
    b.write_shtooka();
}

int64_t CompositeBlock::write_and_get_duration_samples() const {
    const int64_t a_duration = a.write_and_get_duration_samples();
    const int64_t b_duration = b.write_and_get_duration_samples();
    return a_duration + b_duration;
}
string CompositeBlock::blurb() const { return "CompositeBlock(" + a.blurb() + ", " + b.blurb() + ")"; }
