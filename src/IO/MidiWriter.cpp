#include "MidiWriter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include "../Core/Smoketest.h"
#include "../Core/State/GlobalState.h"
#include "Writer.h"

using namespace std;

namespace {

// 120 BPM at 960 ticks per quarter note, so a tick is 1/1920 of a second. Small
// enough resolution that rounding frame times to the nearest tick is imperceptible.
const int ticks_per_quarter_note = 960;
const int microseconds_per_quarter_note = 500000;
const double ticks_per_second = ticks_per_quarter_note * 1000000.0 / microseconds_per_quarter_note;

// Shortest note emitted, so a brief effect is still grabbable in a piano roll.
const long long min_note_ticks = 30;

const int note_base = 60; // Middle C, then one semitone per voice
const int default_velocity = 100;

const uint8_t cc_mod_wheel = 1; // Linked-variable curves

// Ordering for events sharing a tick.
const int order_note_off = 0;
const int order_controller = 1;
const int order_note_on = 2;

struct MidiEvent {
    long long tick;
    int order;
    vector<uint8_t> bytes;
};

int clamp_to_midi_range(long value) {
    if (value < 0) return 0;
    if (value > 127) return 127;
    return static_cast<int>(value);
}

void push_bytes(vector<uint8_t>& out, initializer_list<uint8_t> bytes) {
    out.insert(out.end(), bytes);
}

void push_ascii(vector<uint8_t>& out, const string& text) {
    out.insert(out.end(), text.begin(), text.end());
}

void push_u16(vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value & 0xff));
}

void push_u32(vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(value & 0xff));
}

// Delta times are base-128 big-endian, high bit set on every byte but the last.
void push_varlen(vector<uint8_t>& out, uint32_t value) {
    uint8_t buffer[5];
    int count = 0;
    buffer[count++] = static_cast<uint8_t>(value & 0x7f);
    while (value >>= 7) {
        buffer[count++] = static_cast<uint8_t>((value & 0x7f) | 0x80);
    }
    while (count > 0) out.push_back(buffer[--count]);
}

void push_meta_text(vector<uint8_t>& out, uint8_t meta_type, const string& text) {
    out.push_back(0xff);
    out.push_back(meta_type);
    push_varlen(out, static_cast<uint32_t>(text.size()));
    push_ascii(out, text);
}

// One MTrk chunk: name, optional tempo, then events as delta times.
vector<uint8_t> build_track(const string& name, bool include_tempo, vector<MidiEvent> events) {
    stable_sort(events.begin(), events.end(), [](const MidiEvent& a, const MidiEvent& b) {
        if (a.tick != b.tick) return a.tick < b.tick;
        return a.order < b.order;
    });

    vector<uint8_t> body;

    push_varlen(body, 0);
    push_meta_text(body, 0x03, name);

    if (include_tempo) {
        push_varlen(body, 0);
        push_bytes(body, {0xff, 0x51, 0x03, static_cast<uint8_t>(microseconds_per_quarter_note >> 16), static_cast<uint8_t>((microseconds_per_quarter_note >> 8) & 0xff), static_cast<uint8_t>(microseconds_per_quarter_note & 0xff)});
    }

    long long previous_tick = 0;
    for (const MidiEvent& event : events) {
        push_varlen(body, static_cast<uint32_t>(event.tick - previous_tick));
        body.insert(body.end(), event.bytes.begin(), event.bytes.end());
        previous_tick = event.tick;
    }

    push_varlen(body, 0);
    push_bytes(body, {0xff, 0x2f, 0x00});

    vector<uint8_t> chunk;
    push_ascii(chunk, "MTrk");
    push_u32(chunk, static_cast<uint32_t>(body.size()));
    chunk.insert(chunk.end(), body.begin(), body.end());
    return chunk;
}

MidiEvent control_change_event(long long tick, int channel, uint8_t controller, int value, int order = order_controller) {
    return MidiEvent{tick, order, {static_cast<uint8_t>(0xb0 | channel), controller, static_cast<uint8_t>(value)}};
}

} // namespace

void MidiWriter::add_note(const string& voice, double t_seconds, double duration_seconds) {
    if (!rendering_on()) return; // Match add_sfx: a smoketest emits nothing

    if (t_seconds < 0)
        throw runtime_error("Midi note time was negative: " + to_string(t_seconds) + " seconds.");
    if (duration_seconds < 0)
        throw runtime_error("Midi note duration was negative: " + to_string(duration_seconds) + " seconds.");

    notes.push_back(Note{t_seconds, duration_seconds, voice_index_for(voice)});
}

void MidiWriter::add_continuous(const string& voice, double t_seconds, double duration_seconds) {
    if (!rendering_on()) return; // Match add_sfx: a smoketest emits nothing

    if (t_seconds < 0)
        throw runtime_error("Midi tone time was negative: " + to_string(t_seconds) + " seconds.");
    if (duration_seconds < 0)
        throw runtime_error("Midi tone duration was negative: " + to_string(duration_seconds) + " seconds.");

    tone_slices.push_back(ToneSlice{t_seconds, duration_seconds, voice_index_for(voice)});
}

int MidiWriter::cc_track_index_for(const string& track_name) {
    for (size_t i = 0; i < cc_track_names.size(); i++) {
        if (cc_track_names[i] == track_name) return static_cast<int>(i);
    }
    cc_track_names.push_back(track_name);
    return static_cast<int>(cc_track_names.size()) - 1;
}

void MidiWriter::add_cc(const string& track_name, double t_seconds, double value) {
    if (!rendering_on()) return;

    if (t_seconds < 0)
        throw runtime_error("Midi cc time was negative: " + to_string(t_seconds) + " seconds.");

    cc_samples.push_back(CCSample{t_seconds, value, cc_track_index_for(track_name)});
}

// Capture everything from the global state and write all ccs
void MidiWriter::capture_global_state() {
    if (!rendering_on()) return;

    const double t = get_global_state("t");
    for (const auto& [name, value] : get_all_global_state()) {
        add_cc(name, t, value);
    }
}

vector<MidiWriter::ToneRun> MidiWriter::tone_runs() const {
    vector<ToneSlice> ordered_slices = tone_slices;
    stable_sort(ordered_slices.begin(), ordered_slices.end(), [](const ToneSlice& a, const ToneSlice& b) {
        if (a.voice_index != b.voice_index) return a.voice_index < b.voice_index;
        return a.t_seconds < b.t_seconds;
    });

    vector<ToneRun> runs;
    for (const ToneSlice& slice : ordered_slices) {
        // Adjacent slices continue the same sounding; a real gap starts a fresh note.
        if (!runs.empty() && runs.back().voice_index == slice.voice_index) {
            const ToneSlice& previous = runs.back().slices.back();
            const double previous_end = previous.t_seconds + previous.duration_seconds;
            if (slice.t_seconds - previous_end <= 0.5 * slice.duration_seconds) {
                runs.back().slices.push_back(slice);
                continue;
            }
        }
        runs.push_back(ToneRun{slice.voice_index, vector<ToneSlice>{slice}});
    }
    return runs;
}

int MidiWriter::voice_index_for(const string& voice) {
    const string name = voice.empty() ? "sfx" : voice;
    for (size_t i = 0; i < voice_names.size(); i++) {
        if (voice_names[i] == name) return static_cast<int>(i);
    }
    voice_names.push_back(name);
    return static_cast<int>(voice_names.size()) - 1;
}

int MidiWriter::channel_for(int voice_index) const {
    // Skip channel 10 (index 9), which General MIDI reserves for percussion.
    const int slot = voice_index % 15;
    return slot < 9 ? slot : slot + 1;
}

int MidiWriter::note_number_for(int voice_index) const {
    // Every effect gets its own fixed lane, like a drum map.
    return clamp_to_midi_range(note_base + voice_index);
}

vector<MidiWriter::Note> MidiWriter::notes_in_time_order() const {
    vector<Note> ordered_notes = notes;
    stable_sort(ordered_notes.begin(), ordered_notes.end(), [](const Note& a, const Note& b) {
        return a.t_seconds < b.t_seconds;
    });
    return ordered_notes;
}

void MidiWriter::write_midi_file() const {
    const string midi_path = "io_out/Video.mid";

    vector<vector<MidiEvent>> events_per_voice(voice_names.size());

    // Walked in time order: a repeat of the same note on the same channel must
    // shorten its predecessor, or the older note-off would cut the new note short.
    map<pair<int, int>, pair<int, size_t>> pending_note_offs; // (channel, note) -> (voice, note-off index)

    for (const Note& note : notes_in_time_order()) {
        const int note_number = note_number_for(note.voice_index);
        const int channel = channel_for(note.voice_index);

        const long long start_tick = llround(note.t_seconds * ticks_per_second);
        const long long end_tick = start_tick + max(min_note_ticks, llround(note.duration_seconds * ticks_per_second));

        const auto key = make_pair(channel, note_number);
        const auto pending = pending_note_offs.find(key);
        if (pending != pending_note_offs.end()) {
            vector<MidiEvent>& previous_events = events_per_voice[pending->second.first];
            MidiEvent& previous_on = previous_events[pending->second.second - 1];
            MidiEvent& previous_off = previous_events[pending->second.second];
            if (previous_off.tick > start_tick && start_tick > previous_on.tick) {
                previous_off.tick = start_tick; // Note-offs sort first at equal ticks
            }
        }

        vector<MidiEvent>& events = events_per_voice[note.voice_index];
        events.push_back(MidiEvent{start_tick, order_note_on, {static_cast<uint8_t>(0x90 | channel), static_cast<uint8_t>(note_number), static_cast<uint8_t>(default_velocity)}});
        events.push_back(MidiEvent{end_tick, order_note_off, {static_cast<uint8_t>(0x80 | channel), static_cast<uint8_t>(note_number), 0}});
        pending_note_offs[key] = make_pair(note.voice_index, events.size() - 1);
    }

    // Each tone run becomes one sustained note, held from its first slice's start
    // to its last slice's end, so a DAW shows one continuous held note per run.
    for (const ToneRun& run : tone_runs()) {
        if (run.slices.empty()) continue;

        vector<MidiEvent>& events = events_per_voice[run.voice_index];
        const int channel = channel_for(run.voice_index);
        const int note_number = note_number_for(run.voice_index);

        const ToneSlice& first_slice = run.slices.front();
        const ToneSlice& last_slice = run.slices.back();
        const long long start_tick = llround(first_slice.t_seconds * ticks_per_second);
        const long long end_tick = llround((last_slice.t_seconds + last_slice.duration_seconds) * ticks_per_second);

        events.push_back(MidiEvent{start_tick, order_note_on, {static_cast<uint8_t>(0x90 | channel), static_cast<uint8_t>(note_number), static_cast<uint8_t>(default_velocity)}});
        events.push_back(MidiEvent{end_tick, order_note_off, {static_cast<uint8_t>(0x80 | channel), static_cast<uint8_t>(note_number), 0}});
    }

    // Each linked state variable becomes its own automation-only track: no notes,
    // just a CC curve scaled to that track's own observed min/max range.
    vector<vector<MidiEvent>> events_per_cc_track(cc_track_names.size());
    {
        vector<double> track_min(cc_track_names.size(), 0);
        vector<double> track_max(cc_track_names.size(), 0);
        vector<bool> track_seen(cc_track_names.size(), false);
        for (const CCSample& sample : cc_samples) {
            const int i = sample.track_index;
            if (!track_seen[i]) { track_min[i] = track_max[i] = sample.value; track_seen[i] = true; continue; }
            track_min[i] = min(track_min[i], sample.value);
            track_max[i] = max(track_max[i], sample.value);
        }

        vector<CCSample> ordered_samples = cc_samples;
        stable_sort(ordered_samples.begin(), ordered_samples.end(), [](const CCSample& a, const CCSample& b) {
            if (a.track_index != b.track_index) return a.track_index < b.track_index;
            return a.t_seconds < b.t_seconds;
        });

        vector<int> last_value(cc_track_names.size(), -1);
        for (const CCSample& sample : ordered_samples) {
            const int i = sample.track_index;
            const double range = track_max[i] - track_min[i];
            const int scaled = range > 0 ? clamp_to_midi_range(lround((sample.value - track_min[i]) / range * 127.0)) : 64;
            if (scaled == last_value[i]) continue;
            const long long tick = llround(sample.t_seconds * ticks_per_second);
            events_per_cc_track[i].push_back(control_change_event(tick, channel_for(static_cast<int>(voice_names.size()) + i), cc_mod_wheel, scaled));
            last_value[i] = scaled;
        }
    }

    vector<vector<uint8_t>> tracks;
    tracks.push_back(build_track("swaptube", true, {}));
    for (size_t i = 0; i < voice_names.size(); i++) {
        tracks.push_back(build_track(voice_names[i], false, events_per_voice[i]));
    }
    for (size_t i = 0; i < cc_track_names.size(); i++) {
        tracks.push_back(build_track(cc_track_names[i], false, events_per_cc_track[i]));
    }

    vector<uint8_t> file;
    push_ascii(file, "MThd");
    push_u32(file, 6);
    push_u16(file, 1);
    push_u16(file, static_cast<uint16_t>(tracks.size()));
    push_u16(file, ticks_per_quarter_note);
    for (const vector<uint8_t>& track : tracks) {
        file.insert(file.end(), track.begin(), track.end());
    }

    // Binary, or Windows would expand stray 0x0a bytes to CRLF and corrupt it.
    ofstream midi_file(midi_path, ios::binary);
    if (!midi_file.is_open()) throw runtime_error("Failed to open file: " + midi_path);
    midi_file.write(reinterpret_cast<const char*>(file.data()), static_cast<streamsize>(file.size()));
    midi_file.close();

    cout << "MidiWriter: wrote " << notes.size() << " note(s) and " << tone_runs().size()
         << " tone(s) from " << voice_names.size() << " effect(s), plus " << cc_samples.size()
         << " cc sample(s) from " << cc_track_names.size() << " linked variable(s), across "
         << tracks.size() << " track(s) to " << midi_path << "." << endl;
}

MidiWriter::~MidiWriter() {
    if (is_smoketest()) return; // A smoketest queues no effects

    if (notes.empty() && tone_slices.empty() && cc_samples.empty()) {
        cout << "MidiWriter: no events were queued; skipping MIDI export." << endl;
        return;
    }

    write_midi_file();
}
