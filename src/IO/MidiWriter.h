#pragma once

#include <string>
#include <vector>

// Export a MIDI file given flags passed to go.sh
struct MidiOptions {
    bool enabled = false;      // 'm': write the MIDI file at all
    bool pitch = false;        // 'p': map each effect's frequency onto a note number
    bool velocity = false;     // 'v': map each effect's volume onto a note velocity
    bool split_tracks = false; // 't': give every named effect its own MIDI track

    int bend_range_semitones = 24;

    static MidiOptions parse(const std::string& spec);
    std::string blurb() const;
};

class MidiWriter {
public:
    MidiWriter(const MidiOptions& options);
    ~MidiWriter();

    // Records one sound effect.
    void add_note(const std::string& voice, double t_seconds, double frequency_hz, double duration_seconds, double volume);

    // Records one slice of a continuously sounding tone
    void add_continuous(const std::string& voice, double t_seconds, double duration_seconds, double frequency_hz, double volume);

    bool is_enabled() const;

private:
    struct Note {
        double t_seconds;
        double duration_seconds;
        double frequency_hz;
        double volume;
        int voice_index;
    };

    // One frame's worth of a sustained tone.
    struct ToneSlice {
        double t_seconds;
        double duration_seconds;
        double frequency_hz;
        double volume;
        int voice_index;
    };

    // Consecutive slices that met end to end, i.e. one continuous sounding.
    struct ToneRun {
        int voice_index;
        std::vector<ToneSlice> slices;
    };

    const MidiOptions options;
    std::vector<Note> notes;
    std::vector<ToneSlice> tone_slices;
    std::vector<std::string> voice_names; // Indexed by voice_index, in first-seen order

    int voice_index_for(const std::string& voice);
    int channel_for(int voice_index) const;
    int note_number_for(const Note& note, double& cents_offset) const;
    int velocity_for(const Note& note) const;
    std::vector<Note> notes_in_time_order() const;
    std::vector<ToneRun> tone_runs() const;

    void write_midi_file() const;
    void write_csv_file() const;
};
