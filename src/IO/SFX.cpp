#include "Writer.h"
#include "AudioWriter.h"

void sfx_boink(double time, double freq, double halflife_seconds, double volume){
    double halflife_samples = halflife_seconds * get_audio_samplerate_hz();

    int num_samples = halflife_samples * 5;

    vector<sample_t> left;
    left.reserve(num_samples);

    double sine_multiplier = freq * 6.283 / get_audio_samplerate_hz();

    for(int i = 0; i < num_samples; i++){
        float val_f = pow(.5, i / halflife_samples) * sin(i * sine_multiplier);
        left.push_back(float_to_sample(val_f * 0.07 * volume));
    }

    get_writer().audio->add_sfx(left, left, time);
}

void sfx_clap(double time, double halflife_seconds, double volume){
    double halflife_samples = halflife_seconds * get_audio_samplerate_hz();

    int num_samples = halflife_samples * 5;

    vector<sample_t> left;
    left.reserve(num_samples);

    for(int i = 0; i < num_samples; i++){
        float val_f = pow(.5, i / halflife_samples) * ((rand() / (float)RAND_MAX) * 2 - 1);
        left.push_back(float_to_sample(val_f * 0.07 * volume));
    }

    get_writer().audio->add_sfx(left, left, time);
}
