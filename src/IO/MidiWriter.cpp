#include "MidiWriter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include "../Core/Smoketest.h"

using namespace std;

namespace {

// 120 BPM at 960 ticks per quarter note, so a tick is 1/1920 of a second - every
// framerate swaptube accepts divides that evenly, keeping frames on exact ticks.
const int ticks_per_quarter_note = 960;
const int microseconds_per_quarter_note = 500000;
const double ticks_per_second = ticks_per_quarter_note * 1000000.0 / microseconds_per_quarter_note;

// Shortest note emitted, so a brief effect is still grabbable in a piano roll.
const long long min_note_ticks = 30;

const int unpitched_note_base = 60; // Middle C, then one semitone per voice
const int default_velocity = 100;

const int min_bend_range_semitones = 1;
const int max_bend_range_semitones = 96;

const uint8_t cc_expression = 11; // Volume within a note, as opposed to track level

// Ordering for events sharing a tick, so a note begins already bent and at volume.
const int order_setup = -1;
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
        push_bytes(body, {0xff, 0x51, 0x03,
                          static_cast<uint8_t>(microseconds_per_quarter_note >> 16),
                          static_cast<uint8_t>((microseconds_per_quarter_note >> 8) & 0xff),
                          static_cast<uint8_t>(microseconds_per_quarter_note & 0xff)});
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

double frequency_of_note(int note_number) {
    return 440.0 * pow(2.0, (note_number - 69) / 12.0);
}

double cents_between(double frequency_hz, int note_number) {
    return 1200.0 * log2(frequency_hz / frequency_of_note(note_number));
}

double cents_interval(double from_hz, double to_hz) {
    return 1200.0 * log2(to_hz / from_hz);
}

MidiEvent control_change_event(long long tick, int channel, uint8_t controller, int value, int order = order_controller) {
    return MidiEvent{tick, order, {static_cast<uint8_t>(0xb0 | channel),
                                   controller,
                                   static_cast<uint8_t>(value)}};
}

// 14 bits, centred on 8192, spanning the announced range.
int pitch_bend_value(double cents, double bend_range_cents) {
    const long value = 8192 + lround((cents / bend_range_cents) * 8191.0);
    if (value < 0) return 0;
    if (value > 16383) return 16383;
    return static_cast<int>(value);
}

MidiEvent pitch_bend_event(long long tick, int channel, int value) {
    return MidiEvent{tick, order_controller, {static_cast<uint8_t>(0xe0 | channel),
                                              static_cast<uint8_t>(value & 0x7f),
                                              static_cast<uint8_t>((value >> 7) & 0x7f)}};
}

// RPN 0 is pitch-bend sensitivity. Instruments which ignore it must be set by hand.
void push_bend_range_setup(vector<MidiEvent>& events, long long tick, int channel, int bend_range_semitones) {
    events.push_back(control_change_event(tick, channel, 101, 0, order_setup));
    events.push_back(control_change_event(tick, channel, 100, 0, order_setup));
    events.push_back(control_change_event(tick, channel, 6, bend_range_semitones, order_setup));
    events.push_back(control_change_event(tick, channel, 38, 0, order_setup));
    events.push_back(control_change_event(tick, channel, 101, 127, order_setup)); // RPN null, so a
    events.push_back(control_change_event(tick, channel, 100, 127, order_setup)); // later CC6 is inert
}

string csv_quote(const string& text) {
    string quoted = "\"";
    for (char c : text) {
        if (c == '"') quoted += '"';
        quoted += c;
    }
    return quoted + "\"";
}

} // namespace

MidiOptions MidiOptions::parse(const string& spec) {
    MidiOptions options;
    if (spec.empty() || spec == "-" || spec == "none") return options;

    string bend_range_digits;
    for (char c : spec) {
        switch (c) {
            case 'm': options.enabled      = true; break;
            case 'p': options.pitch        = true; break;
            case 'v': options.velocity     = true; break;
            case 't': options.split_tracks = true; break;
            default:
                if (c >= '0' && c <= '9') { bend_range_digits += c; break; }
                throw runtime_error("Invalid midi option '" + string(1, c) + "' in spec \"" + spec +
                                    "\". Expected some combination of m, p, v and t, optionally"
                                    " followed by the pitch bend range in semitones.");
        }
    }

    if (!bend_range_digits.empty()) {
        options.bend_range_semitones = stoi(bend_range_digits);
        if (options.bend_range_semitones < min_bend_range_semitones ||
            options.bend_range_semitones > max_bend_range_semitones) {
            throw runtime_error("Midi pitch bend range must be between " + to_string(min_bend_range_semitones) +
                                " and " + to_string(max_bend_range_semitones) + " semitones, but \"" + spec +
                                "\" asked for " + to_string(options.bend_range_semitones) + ".");
        }
    }

    options.enabled = true; // Any sub-option stands in for 'm'
    return options;
}

string MidiOptions::blurb() const {
    if (!enabled) return "off";
    string description = split_tracks ? "track per effect" : "single track";
    description += pitch ? ", pitched" : ", fixed pitch";
    description += velocity ? ", velocity from volume" : ", fixed velocity";
    if (pitch) description += ", bend range " + to_string(bend_range_semitones) + " semitones";
    return description;
}

MidiWriter::MidiWriter(const MidiOptions& options) : options(options) {}

bool MidiWriter::is_enabled() const { return options.enabled; }

void MidiWriter::add_note(const string& voice, double t_seconds, double frequency_hz, double duration_seconds, double volume) {
    if (!options.enabled) return;
    if (!rendering_on()) return; // Match add_sfx: a smoketest emits nothing

    if (t_seconds < 0)
        throw runtime_error("Midi note time was negative: " + to_string(t_seconds) + " seconds.");
    if (duration_seconds < 0)
        throw runtime_error("Midi note duration was negative: " + to_string(duration_seconds) + " seconds.");

    notes.push_back(Note{t_seconds, duration_seconds, frequency_hz, volume, voice_index_for(voice)});
}

void MidiWriter::add_continuous(const string& voice, double t_seconds, double duration_seconds, double frequency_hz, double volume) {
    if (!options.enabled) return;
    if (!rendering_on()) return; // Match add_sfx: a smoketest emits nothing

    if (t_seconds < 0)
        throw runtime_error("Midi tone time was negative: " + to_string(t_seconds) + " seconds.");
    if (duration_seconds < 0)
        throw runtime_error("Midi tone duration was negative: " + to_string(duration_seconds) + " seconds.");

    tone_slices.push_back(ToneSlice{t_seconds, duration_seconds, frequency_hz, volume, voice_index_for(voice)});
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

int MidiWriter::note_number_for(const Note& note, double& cents_offset) const {
    cents_offset = 0;

    // Without pitch export, each effect still gets its own lane, like a drum map.
    if (!options.pitch || note.frequency_hz <= 0)
        return clamp_to_midi_range(unpitched_note_base + note.voice_index);

    const double exact = 69.0 + 12.0 * log2(note.frequency_hz / 440.0);
    const int rounded = clamp_to_midi_range(lround(exact));
    cents_offset = (exact - rounded) * 100.0;
    return rounded;
}

int MidiWriter::velocity_for(const Note& note) const {
    if (!options.velocity) return default_velocity;
    // Volume 1 lands on the default, leaving headroom for effects which overshoot.
    // Velocity 0 would read as a note-off, so the quietest effect still gets a 1.
    return max(1, clamp_to_midi_range(lround(note.volume * default_velocity)));
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

    // Walked in time order: a repeat of the same pitch on the same channel must
    // shorten its predecessor, or the older note-off would cut the new note short.
    map<pair<int, int>, pair<int, size_t>> pending_note_offs; // (channel, note) -> (voice, note-off index)

    for (const Note& note : notes_in_time_order()) {
        double cents_offset = 0;
        const int note_number = note_number_for(note, cents_offset);
        const int velocity = velocity_for(note);
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
        events.push_back(MidiEvent{start_tick, order_note_on, {static_cast<uint8_t>(0x90 | channel),
                                                               static_cast<uint8_t>(note_number),
                                                               static_cast<uint8_t>(velocity)}});
        events.push_back(MidiEvent{end_tick, order_note_off, {static_cast<uint8_t>(0x80 | channel),
                                                              static_cast<uint8_t>(note_number),
                                                              0}});
        pending_note_offs[key] = make_pair(note.voice_index, events.size() - 1);
    }

    // Each tone run becomes one sustained note, with its pitch and volume traced
    // by controller curves, so a DAW shows a held note which glides and swells.
    set<int> channels_with_bend_setup;
    const double bend_range_cents = options.bend_range_semitones * 100.0;

    for (const ToneRun& run : tone_runs()) {
        if (run.slices.empty()) continue;

        vector<MidiEvent>& events = events_per_voice[run.voice_index];
        const int channel = channel_for(run.voice_index);

        if (options.pitch && channels_with_bend_setup.insert(channel).second) {
            push_bend_range_setup(events, 0, channel, options.bend_range_semitones);
        }

        const ToneSlice& first_slice = run.slices.front();

        double lowest = 0;
        double highest = 0;
        double peak_volume = 0;
        for (const ToneSlice& slice : run.slices) {
            peak_volume = max(peak_volume, slice.volume);
            if (slice.frequency_hz <= 0) continue;
            if (lowest == 0 || slice.frequency_hz < lowest) lowest = slice.frequency_hz;
            if (slice.frequency_hz > highest) highest = slice.frequency_hz;
        }

        // Anchor on the pitch the tone begins at, so the note reads true even in an
        // instrument whose bend range was never set. Only a glide too wide for the
        // bend anchors at the middle instead, buying twice the span.
        double anchor_frequency = first_slice.frequency_hz;
        if (lowest > 0 && first_slice.frequency_hz > 0) {
            const double reach_cents = max(fabs(cents_interval(first_slice.frequency_hz, highest)),
                                           fabs(cents_interval(first_slice.frequency_hz, lowest)));
            if (reach_cents > bend_range_cents) anchor_frequency = sqrt(lowest * highest);
        }

        double unused_cents = 0;
        int base_note = note_number_for(Note{first_slice.t_seconds, first_slice.duration_seconds,
                                             anchor_frequency, first_slice.volume,
                                             first_slice.voice_index}, unused_cents);

        // Velocity stands for the whole run, so it takes the peak rather than the
        // fade-in, which is near silence. Expression below traces the shape as a
        // fraction of that peak, so the two together reproduce the envelope.
        const int run_velocity = velocity_for(Note{first_slice.t_seconds, first_slice.duration_seconds,
                                                   first_slice.frequency_hz, peak_volume,
                                                   first_slice.voice_index});

        int last_bend_value = -1;   // -1 means nothing emitted yet
        int last_expression = -1;
        long long run_end_tick = llround(first_slice.t_seconds * ticks_per_second);

        auto open_note = [&](long long tick, int note_number, int velocity) {
            events.push_back(MidiEvent{tick, order_note_on, {static_cast<uint8_t>(0x90 | channel),
                                                             static_cast<uint8_t>(note_number),
                                                             static_cast<uint8_t>(velocity)}});
        };
        auto close_note = [&](long long tick, int note_number) {
            events.push_back(MidiEvent{tick, order_note_off, {static_cast<uint8_t>(0x80 | channel),
                                                              static_cast<uint8_t>(note_number),
                                                              0}});
        };
        auto emit_bend = [&](long long tick, double cents) {
            const int value = pitch_bend_value(cents, bend_range_cents);
            if (value == last_bend_value) return;
            events.push_back(pitch_bend_event(tick, channel, value));
            last_bend_value = value;
        };
        auto emit_expression = [&](long long tick, double volume) {
            const double fraction = peak_volume > 0 ? volume / peak_volume : 0;
            const int value = clamp_to_midi_range(lround(fraction * 127.0));
            if (value == last_expression) return;
            events.push_back(control_change_event(tick, channel, cc_expression, value));
            last_expression = value;
        };

        for (size_t i = 0; i < run.slices.size(); i++) {
            const ToneSlice& slice = run.slices[i];
            const Note as_note{slice.t_seconds, slice.duration_seconds, slice.frequency_hz,
                               slice.volume, slice.voice_index};
            const long long tick = llround(slice.t_seconds * ticks_per_second);
            run_end_tick = llround((slice.t_seconds + slice.duration_seconds) * ticks_per_second);

            const bool bendable = options.pitch && slice.frequency_hz > 0;
            double cents = bendable ? cents_between(slice.frequency_hz, base_note) : 0;

            // A pitch past what the bend can reach is restated on a nearer note,
            // rather than silently flattening against the limit.
            const bool reanchoring = i > 0 && bendable && fabs(cents) > bend_range_cents;
            if (reanchoring) {
                close_note(tick, base_note);
                base_note = note_number_for(as_note, unused_cents);
                cents = cents_between(slice.frequency_hz, base_note);
                last_bend_value = -1;
            }

            if (bendable) emit_bend(tick, cents);
            if (options.velocity) emit_expression(tick, slice.volume);
            if (i == 0 || reanchoring) open_note(tick, base_note, run_velocity);
        }

        close_note(run_end_tick, base_note);

        // Leave the channel unbent, so whatever the user puts on this track
        // afterwards is not detuned by a leftover bend.
        if (options.pitch && last_bend_value != 8192 && last_bend_value != -1) {
            events.push_back(pitch_bend_event(run_end_tick, channel, 8192));
        }
    }

    vector<vector<uint8_t>> tracks;
    if (options.split_tracks) {
        tracks.push_back(build_track("swaptube", true, {}));
        for (size_t i = 0; i < voice_names.size(); i++) {
            tracks.push_back(build_track(voice_names[i], false, events_per_voice[i]));
        }
    } else {
        vector<MidiEvent> all_events;
        for (const vector<MidiEvent>& events : events_per_voice) {
            all_events.insert(all_events.end(), events.begin(), events.end());
        }
        tracks.push_back(build_track("swaptube sfx", true, all_events));
    }

    vector<uint8_t> file;
    push_ascii(file, "MThd");
    push_u32(file, 6);
    push_u16(file, options.split_tracks ? 1 : 0);
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
         << " tone(s) from " << voice_names.size() << " effect(s) across " << tracks.size()
         << " track(s) to " << midi_path << "." << endl;

    // Most DAWs ignore the RPN which announces this, so it must be set by hand
    // or every glide plays out of tune by the ratio of the two numbers.
    if (options.pitch && !tone_slices.empty()) {
        cout << "MidiWriter: SET YOUR INSTRUMENT'S PITCH BEND RANGE TO "
             << options.bend_range_semitones << " SEMITONES, or continuous tones will play out of"
             << " tune. Pass a different -b to match an instrument that cannot reach that." << endl;
    }
}

void MidiWriter::write_csv_file() const {
    // The MIDI rounds every effect to a semitone; this sidecar keeps the exact
    // frequencies and volumes the render asked for.
    const string csv_path = "io_out/Video.sfx.csv";

    ofstream csv_file(csv_path);
    if (!csv_file.is_open()) throw runtime_error("Failed to open file: " + csv_path);

    csv_file << "t_seconds,kind,voice,frequency_hz,duration_seconds,volume,"
                "midi_track,midi_channel,midi_note,cents,velocity,expression\n";

    // Notes and tone slices, interleaved in time order.
    vector<pair<double, string>> rows;

    for (const Note& note : notes) {
        double cents_offset = 0;
        const int note_number = note_number_for(note, cents_offset);
        ostringstream row;
        row << fixed
            << setprecision(6) << note.t_seconds << ",note,"
            << csv_quote(voice_names[note.voice_index]) << ","
            << setprecision(4) << note.frequency_hz << ","
            << setprecision(6) << note.duration_seconds << ","
            << setprecision(4) << note.volume << ","
            // Numbered as a DAW displays them: tracks after the tempo track,
            // channels from one.
            << (options.split_tracks ? note.voice_index + 1 : 0) << ","
            << channel_for(note.voice_index) + 1 << ","
            << note_number << ","
            << setprecision(1) << cents_offset << ","
            << velocity_for(note) << ","; // No expression; a note carries no curve
        rows.push_back(make_pair(note.t_seconds, row.str()));
    }

    // Run by run, since a slice's velocity and expression are both relative to
    // the peak of the run it belongs to.
    for (const ToneRun& run : tone_runs()) {
        double peak_volume = 0;
        for (const ToneSlice& slice : run.slices) peak_volume = max(peak_volume, slice.volume);

        const ToneSlice& first_slice = run.slices.front();
        const int run_velocity = velocity_for(Note{first_slice.t_seconds, first_slice.duration_seconds,
                                                   first_slice.frequency_hz, peak_volume,
                                                   first_slice.voice_index});

        for (const ToneSlice& slice : run.slices) {
            const double fraction = peak_volume > 0 ? slice.volume / peak_volume : 0;
            ostringstream row;
            row << fixed
                << setprecision(6) << slice.t_seconds << ",tone,"
                << csv_quote(voice_names[slice.voice_index]) << ","
                << setprecision(4) << slice.frequency_hz << ","
                << setprecision(6) << slice.duration_seconds << ","
                << setprecision(4) << slice.volume << ","
                << (options.split_tracks ? slice.voice_index + 1 : 0) << ","
                << channel_for(slice.voice_index) + 1
                // A tone's pitch is a bend from its run's anchor, which no single
                // column can hold, so these stay empty and frequency_hz is the truth.
                << ",,,"
                << run_velocity << ","
                << clamp_to_midi_range(lround(fraction * 127.0));
            rows.push_back(make_pair(slice.t_seconds, row.str()));
        }
    }

    stable_sort(rows.begin(), rows.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
        return a.first < b.first;
    });

    for (const pair<double, string>& row : rows) csv_file << row.second << "\n";

    csv_file.close();

    cout << "MidiWriter: wrote " << rows.size() << " event(s) to " << csv_path << "." << endl;
}

MidiWriter::~MidiWriter() {
    if (!options.enabled) return;
    if (is_smoketest()) return; // A smoketest queues no effects

    if (notes.empty() && tone_slices.empty()) {
        cout << "MidiWriter: no sound effects were queued; skipping MIDI export." << endl;
        return;
    }

    write_midi_file();
    write_csv_file();
}
