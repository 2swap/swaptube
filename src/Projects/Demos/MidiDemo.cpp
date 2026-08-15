#include "../Scenes/Media/MidiSoundScene.h"
#include "../IO/Writer.h"

// Pentatonic
const vector<double> scale = {220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00};

void render_video() {
    MidiSoundScene scene;
    // This is now the syntax to export MIDI
    scene.stage_publish_to_global = {{"drone_frequency", "drone_frequency"}, {"drone_volume", "drone_volume"}};

    stage_macroblock(SilenceBlock(1), 1);
    scene.render_microblock();

    // Discrete notes, rising in pitch and volume.
    stage_macroblock(SilenceBlock(4), 1);
    double t = get_global_state("t");
    for (size_t i = 0; i < scale.size(); i++) {
        const double when = t + 0.35 * i;
        const double volume = 0.2 + 0.8 * i / (scale.size() - 1.0);
        scene.play_note(when, scale[i], volume);
        // This is the syntax to add a discrete MIDI note
        get_writer().midi->add_note("beat", when, 0);
    }
    scene.render_microblock();

    // A drone that fades in, glides up a whole tone, and fades out.
    stage_macroblock(SilenceBlock(2), 2);
    scene.manager.set("drone_frequency", "330");
    scene.manager.transition(MICRO, "drone_volume", "1");
    scene.render_microblock();
    scene.manager.transition(MICRO, "drone_frequency", "370");
    scene.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    scene.manager.transition(MICRO, "drone_volume", "0");
    scene.render_microblock();

    // A two-octave glide
    stage_macroblock(SilenceBlock(1), 1);
    scene.manager.set("drone_frequency", "220");
    scene.manager.transition(MICRO, "drone_volume", "0.9");
    scene.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    scene.manager.transition(MICRO, "drone_frequency", "880");
    scene.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    scene.manager.transition(MICRO, "drone_frequency", "220");
    scene.manager.transition(MICRO, "drone_volume", "0");
    scene.render_microblock();

    // Both together: a drone underneath, notes on top
    stage_macroblock(SilenceBlock(1), 1);
    scene.manager.set("drone_frequency", "220");
    scene.manager.transition(MICRO, "drone_volume", "0.7");
    scene.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    t = get_global_state("t");
    for (int i = 0; i < 12; i++) {
        scene.play_note(t + 0.4 * i, scale[(i * 3) % scale.size()], 0.5 + 0.4 * ((i % 3) == 0));
    }
    scene.manager.transition(MICRO, "drone_frequency", "293.66");
    scene.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    scene.manager.transition(MICRO, "drone_volume", "0");
    scene.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    scene.render_microblock();
}
