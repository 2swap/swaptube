#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

// Define a struct to hold a single sine wave's parameters
struct SineWave {
    double frequency; // in Hz
    double amplitude;
};

class FourierSound {
public:
    // Vector containing multiple sine waves that will be summed to create the sound
    vector<SineWave> sine_waves;

    // Helper function to add a sine wave to the FourierSound object
    void add_sine_wave(double frequency, double amplitude) {
        sine_waves.push_back({frequency, amplitude});
    }

    // File format:
    //   <number_of_sine_waves>
    //   <frequency> <amplitude>
    //   ...
    bool save(const string& filename) const {
        ofstream ofs(PATH_MANAGER.sfx_dir + filename);
        if (!ofs) {
            throw runtime_error("Error opening file for writing: " + filename);
        }
        ofs << sine_waves.size() << "\n";
        for (const auto& sine : sine_waves) {
            ofs << sine.frequency << " " << sine.amplitude << " " << "\n";
        }
        return true;
    }

    FourierSound(const string& filename) {
        ifstream ifs(PATH_MANAGER.sfx_dir + filename);
        if (!ifs) {
            throw runtime_error("Error opening file for reading: " + filename);
        }
        size_t count;
        ifs >> count;
        sine_waves.clear();
        for (size_t i = 0; i < count; ++i) {
            SineWave sine;
            ifs >> sine.frequency >> sine.amplitude;
            sine_waves.push_back(sine);
        }
    }
};

const double two_pi = 2.0 * M_PI;

class Soundscape {
public:
    vector<FourierSound> effects;
    vector<double> integrators;
    vector<double> frequencies;

    void add(const FourierSound& fs) {
        effects.push_back(fs);
        integrators.push_back(0);
        frequencies.push_back(1);
    }

    void generate_audio(int total_samples, vector<float>& left, vector<float>& right, const vector<double>& new_frequencies) {
        if (new_frequencies.size() != frequencies.size()) throw runtime_error("ERROR: frequency counts do not match!");

        const int sample_rate = 44100;

        // Reserve space for efficiency
        left.reserve(total_samples);
        right.reserve(total_samples);

        // Iterate through each sample index
        for (int i = 0; i < total_samples; ++i) {
            double w = i / static_cast<double>(total_samples);

            float sample_value = 0.0;
            for (int e = 0; e < effects.size(); e++) {
                const FourierSound& fs = effects[e];
                for (const SineWave& sw : fs.sine_waves) {
                    sample_value += sw.amplitude * sin(integrators[e] * sw.frequency);
                }
                integrators[e] += two_pi * lerp(frequencies[e], new_frequencies[e], w) / sample_rate;
            }

            sample_value *= 0.03;
            // Write the resulting sample value to both the left and right channels
            left.push_back(sample_value);
            right.push_back(sample_value);
        }
        frequencies = new_frequencies;
    }
};
