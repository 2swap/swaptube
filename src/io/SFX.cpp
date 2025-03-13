
/*
    void generate_beep(double duration){
        vector<float> left;
        vector<float> right;
        int total_samples = duration*44100;
        double note = state["tone"];
        for(int i = 0; i < total_samples; i++){
            double pct_complete = i/static_cast<double>(total_samples);
            float val = .03*sin(i*note*2200./44100.)/note;
            val *= pow(.5, 4*pct_complete);
            left.push_back(val);
            right.push_back(val);
        }
        WRITER.add_sfx(left, right, state["t"]*44100);
    }
*/

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
    double phase;     // in radians
};

class FourierSound {
public:
    // Vector containing multiple sine waves that will be summed to create the sound
    vector<SineWave> sine_waves;

    // Helper function to add a sine wave to the FourierSound object
    void add_sine_wave(double frequency, double amplitude, double phase) {
        sine_waves.push_back({frequency, amplitude, phase});
    }

    // File format:
    //   <number_of_sine_waves>
    //   <frequency> <amplitude> <phase>
    //   ...
    bool save(const string& filename) const {
        ofstream ofs(PATH_MANAGER.sfx_dir + filename);
        if (!ofs) {
            throw runtime_error("Error opening file for writing: " + filename);
        }
        ofs << sine_waves.size() << "\n";
        for (const auto& sine : sine_waves) {
            ofs << sine.frequency << " " << sine.amplitude << " " << sine.phase << "\n";
        }
        return true;
    }

    bool load(const string& filename) {
        ifstream ifs(PATH_MANAGER.sfx_dir + filename);
        if (!ifs) {
            throw runtime_error("Error opening file for reading: " + filename);
        }
        size_t count;
        ifs >> count;
        sine_waves.clear();
        for (size_t i = 0; i < count; ++i) {
            SineWave sine;
            ifs >> sine.frequency >> sine.amplitude >> sine.phase;
            sine_waves.push_back(sine);
        }
        return true;
    }
};

void generate_audio(double duration, vector<float>& left, vector<float>& right, FourierSound fs1, FourierSound fs2) {
    int num_sins = fs1.sine_waves.size();
    if(num_sins != fs2.sine_waves.size())
	throw runtime_error("ERROR: Sine wave count in fourier series do not match!");

    const int sample_rate = 44100;
    int total_samples = static_cast<int>(duration * sample_rate);

    // Reserve space for efficiency
    left.reserve(total_samples);
    right.reserve(total_samples);

    const double two_pi = 2.0 * M_PI;
    vector<double> timekeepers(num_sins);

    // Iterate through each sample index
    for (int i = 0; i < total_samples; ++i) {
	double w = i / static_cast<double>(total_samples);

        float sample_value = 0.0;
        for (int s = 0; s < num_sins; s++) {
	    const SineWave& s1 = fs1.sine_waves[s];
	    const SineWave& s2 = fs2.sine_waves[s];
            sample_value += smoothlerp(s1.amplitude, s2.amplitude, w) * sin(timekeepers[s] + smoothlerp(s1.phase, s2.phase, w));
	    timekeepers[s] += two_pi * smoothlerp(s1.frequency, s2.frequency, w) / sample_rate;
        }

        sample_value *= .1 * pow(.5, 8*w);
        // Write the resulting sample value to both the left and right channels
        left.push_back(sample_value);
        right.push_back(sample_value);
    }
}
