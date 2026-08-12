#pragma once

#include <string>
#include <vector>

// MIDI export configuration, set via configure_midi(). MIDI export itself is always on;
// this just controls whether the CSV sidecar also gets written.
struct MidiOptions {
    bool csv = false; // also write the io_out/Video.sfx.csv sidecar
};

// Called once, typically at the top of render_video(), to configure MIDI export detail.
void configure_midi(bool csv = false);

class MidiWriter {
public:
    MidiWriter();
    ~MidiWriter();

    void configure(bool csv);

    // Records one discrete event on its own named track.
    void add_note(const std::string& voice, double t_seconds, double duration_seconds);

    // Records one slice of an ongoing event; adjacent slices merge into one held note.
    void add_continuous(const std::string& voice, double t_seconds, double duration_seconds);

    // Records one sample of a state variable's value, for a standalone CC automation track.
    void add_cc(const std::string& track_name, double t_seconds, double value);

private:
    struct Note {
        double t_seconds;
        double duration_seconds;
        int voice_index;
    };

    // One frame's worth of an ongoing event.
    struct ToneSlice {
        double t_seconds;
        double duration_seconds;
        int voice_index;
    };

    // Consecutive slices that met end to end, i.e. one continuous sounding.
    struct ToneRun {
        int voice_index;
        std::vector<ToneSlice> slices;
    };

    struct CCSample {
        double t_seconds;
        double value;
        int track_index;
    };

    MidiOptions options;
    std::vector<Note> notes;
    std::vector<ToneSlice> tone_slices;
    std::vector<std::string> voice_names; // Indexed by voice_index, in first-seen order

    std::vector<std::string> cc_track_names; // Separate namespace from voice_names
    std::vector<CCSample> cc_samples;
    int cc_track_index_for(const std::string& track_name);

    int voice_index_for(const std::string& voice);
    int channel_for(int index) const;
    int note_number_for(int voice_index) const;
    std::vector<Note> notes_in_time_order() const;
    std::vector<ToneRun> tone_runs() const;

    void write_midi_file() const;
    void write_csv_file() const;
};
