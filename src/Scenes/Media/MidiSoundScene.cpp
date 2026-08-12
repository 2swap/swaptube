#include "MidiSoundScene.h"

#include <cmath>
#include <vector>
#include "../../Host_Device_Shared/Color.h"
#include "../../IO/AudioWriter.h"
#include "../../IO/MidiWriter.h"
#include "../../IO/SFX.h"
#include "../../IO/Writer.h"

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);
extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);

namespace {

const double two_pi = 6.283185307179586;

// How much history the trace shows, and the pitch range it spans.
const double trace_window_seconds = 4.0;
const double lowest_plotted_hz = 110.0;
const double highest_plotted_hz = 1760.0; // Four octaves above

// A note's ring keeps expanding and fading for this long after it sounds.
const double note_bloom_seconds = 0.6;

double pitch_fraction(double frequency_hz) {
    if (frequency_hz <= 0) return 0;
    const double low = log2(lowest_plotted_hz);
    const double high = log2(highest_plotted_hz);
    return clamp((log2(frequency_hz) - low) / (high - low), 0.0, 1.0);
}

}

MidiSoundScene::MidiSoundScene(const vec2& dimensions) : Scene(dimensions) {
    manager.set({{"drone_frequency", "220"}, {"drone_volume", "0"}});
    link_cc("drone_frequency");
    link_cc("drone_volume");
}

void MidiSoundScene::play_note(double t_seconds, double frequency_hz, double volume, const std::string& voice) {
    // sfx_boink already does both halves: it sounds the note and hands it to the
    // MIDI exporter.
    sfx_boink(t_seconds, frequency_hz, 0.06, volume, voice);
    note_marks.push_back(Mark{t_seconds, frequency_hz, volume});
}

void MidiSoundScene::emit_drone() {
    const double volume = state["drone_volume"];
    const double frequency = state["drone_frequency"];
    const double t = get_global_state("t");
    const int samplerate = get_audio_samplerate_hz();
    const int samples = get_samples_per_frame();

    if (volume < 0.001 || frequency <= 0) {
        // Silent: let the phase go, so the next sounding starts a fresh run in
        // the exporter rather than continuing this one.
        drone_phase = 0;
        return;
    }

    std::vector<sample_t> channel;
    channel.reserve(samples);

    const double phase_step = two_pi * frequency / samplerate;
    for (int i = 0; i < samples; i++) {
        channel.push_back(float_to_sample(0.08 * volume * sin(drone_phase)));
        drone_phase += phase_step;
    }
    drone_phase = fmod(drone_phase, two_pi); // Keep precision from drifting over a long render

    get_writer().audio->add_sfx(channel, channel, t);
    get_writer().midi->add_continuous("drone", t, samples / static_cast<double>(samplerate));

    drone_trace.push_back(Mark{t, frequency, volume});
}

void MidiSoundScene::draw() {
    emit_drone();

    const double now = get_global_state("t");
    const double window_start = now - trace_window_seconds;
    const int w = get_width();
    const int h = get_height();
    const ivec2 wh = get_width_height();

    // Drop anything that has scrolled off the left edge.
    while (!drone_trace.empty() && drone_trace.front().t_seconds < window_start) drone_trace.pop_front();
    while (!note_marks.empty() && note_marks.front().t_seconds < window_start) note_marks.pop_front();

    auto x_of = [&](double t_seconds) { return static_cast<float>(w * (t_seconds - window_start) / trace_window_seconds); };
    auto y_of = [&](double frequency_hz) { return static_cast<float>(h * (0.92 - 0.84 * pitch_fraction(frequency_hz))); };

    // Octave guide lines, so the pitch axis means something to the eye.
    for (double frequency = lowest_plotted_hz; frequency <= highest_plotted_hz + 1; frequency *= 2) {
        const int y = static_cast<int>(y_of(frequency));
        draw_rectangle(gpu_pix.get_ptr(), wh, ivec2(0, y), ivec2(w, y + 1), argb(70, 255, 255, 255));
    }

    // pitch as height, volume as thickness, colour by pitch.
    for (const Mark& point : drone_trace) {
        const float radius = 1.5f + 9.0f * static_cast<float>(clamp(point.volume, 0.0, 1.0)) * h / 540.0f;
        draw_circle(gpu_pix.get_ptr(), wh, vec2(x_of(point.t_seconds), y_of(point.frequency_hz)), radius, rainbow(pitch_fraction(point.frequency_hz)));
    }

    for (const Mark& mark : note_marks) {
        const double age = now - mark.t_seconds;
        if (age < 0) continue;

        const vec2 center(x_of(mark.t_seconds), y_of(mark.frequency_hz));
        const float loudness = static_cast<float>(clamp(mark.volume, 0.0, 1.0));

        if (age < note_bloom_seconds) {
            const double bloom = age / note_bloom_seconds;
            draw_circle(gpu_pix.get_ptr(), wh, center,
                        static_cast<float>((5.0 + 45.0 * bloom) * h / 540.0),
                        argb(static_cast<int>(200 * (1.0 - bloom) * loudness), 255, 255, 255));
        }

        draw_circle(gpu_pix.get_ptr(), wh, center, (3.0f + 8.0f * loudness) * h / 540.0f, OPAQUE_WHITE);
    }

    // Playhead at the right edge, where the newest sound lands.
    draw_rectangle(gpu_pix.get_ptr(), wh, ivec2(w - 2, 0), ivec2(w, h), argb(120, 255, 255, 255));
}
