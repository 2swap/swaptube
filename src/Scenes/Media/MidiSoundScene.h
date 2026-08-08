#pragma once

#include <deque>
#include <string>
#include "../Scene.h"

// A nothing-special scene that can play notes and a drone and renderes a very simple visual on-screen. The sound is the main thing here.
class MidiSoundScene : public Scene {
public:
    MidiSoundScene(const vec2& dimensions = vec2(1, 1));

    const StateQuery populate_state_query() const override;
    void draw() override;

    bool needs_redraw() const override { return true; }

    // Sounds a discrete note at an absolute time and records it for the trace.
    void play_note(double t_seconds, double frequency_hz, double volume, const std::string& voice = "melody");

private:
    struct Mark {
        double t_seconds;
        double frequency_hz;
        double volume;
    };

    void emit_drone();

    std::deque<Mark> drone_trace;
    std::deque<Mark> note_marks;

    double drone_phase = 0;
};
