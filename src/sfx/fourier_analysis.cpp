#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>

using namespace std;

const double two_pi = 2.0 * M_PI;
const int FREQUENCY_STEP = 1; // Record every nth frequency
const float FREQUENCY_KEEP_RATIO = 0.1f; // Keep top 10% frequencies (discard the 90% with lowest amplitudes)

// Dummy PATH_MANAGER (set directory to current directory)
struct PATH_MANAGER {
    static const string sfx_dir;
};
const string PATH_MANAGER::sfx_dir = "";

// Struct to hold a single sine wave's parameters (used in additive synthesis)
struct SineWave {
    double frequency; // in Hz
    double amplitude;
    double phase;
};

// FourierSound class that stores multiple sine waves and writes them in the expected format.
class FourierSound {
public:
    // For analysis file storage: we store sample_rate and sine wave parameters.
    vector<SineWave> sine_waves;
    int sample_rate = 0;
    int N = 0; // total number of samples used for forward analysis (not saved to file)

    // Add a sine wave to the FourierSound object.
    void add_sine_wave(double frequency, double amplitude, double phase) {
        sine_waves.push_back({frequency, amplitude, phase});
    }

    // Save analysis file.
    // File format:
    //   <sample_rate> <num_sine_waves>
    //   <frequency> <amplitude> <phase>
    //   ...
    bool save(const string& filename) const {
        ofstream ofs(PATH_MANAGER::sfx_dir + filename);
        if (!ofs) {
            throw runtime_error("Error opening file for writing: " + filename);
        }
        ofs << sample_rate << " " << sine_waves.size() << "\n";
        for (size_t i = 0; i < sine_waves.size(); i++) {
            ofs << sine_waves[i].frequency << " " << sine_waves[i].amplitude << " " << sine_waves[i].phase << "\n";
        }
        return true;
    }
};

// --- WAV file reading functions ---
// A minimal WAV header parser. (This code expects 16-bit PCM mono data.)
struct WAVHeader {
    char chunkID[4];     // "RIFF"
    uint32_t chunkSize;
    char format[4];      // "WAVE"
};

struct FMTSubchunk {
    char subchunk1ID[4]; // "fmt "
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

struct DataSubchunk {
    char subchunk2ID[4]; // "data"
    uint32_t subchunk2Size;
};

// Reads a WAV file and outputs normalized samples in the range [-1, 1].
bool read_wav(const string& filename, vector<double>& samples, int& sample_rate) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error opening WAV file: " << filename << "\n";
        return false;
    }
    
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if (strncmp(header.chunkID, "RIFF", 4) != 0 || strncmp(header.format, "WAVE", 4) != 0) {
        cerr << "Not a valid WAV file.\n";
        return false;
    }
    
    FMTSubchunk fmt;
    file.read(reinterpret_cast<char*>(&fmt), sizeof(FMTSubchunk));
    if (strncmp(fmt.subchunk1ID, "fmt ", 4) != 0) {
        cerr << "Invalid fmt subchunk.\n";
        return false;
    }
    
    sample_rate = fmt.sampleRate;
    if (fmt.bitsPerSample != 16) {
        cerr << "Only 16-bit WAV files are supported.\n";
        return false;
    }
    
    // Skip any extra bytes in the fmt subchunk if needed.
    if (fmt.subchunk1Size > 16) {
        file.seekg(fmt.subchunk1Size - 16, ios::cur);
    }
    
    // Look for the "data" subchunk.
    DataSubchunk dataSubchunk;
    while (true) {
        file.read(reinterpret_cast<char*>(&dataSubchunk), sizeof(DataSubchunk));
        if (strncmp(dataSubchunk.subchunk2ID, "data", 4) == 0) {
            break;
        } else {
            // Skip the unknown chunk.
            file.seekg(dataSubchunk.subchunk2Size, ios::cur);
        }
        if (file.eof()) {
            cerr << "Data chunk not found.\n";
            return false;
        }
    }
    
    int num_samples = dataSubchunk.subchunk2Size / (fmt.numChannels * (fmt.bitsPerSample / 8));
    samples.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
        int16_t sample;
        file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
        // Since we forced mono (-ac 1) with ffmpeg, we expect one channel.
        samples[i] = sample / 32768.0;
    }
    return true;
}

// Write a WAV file from normalized samples in the range [-1,1] using 16-bit PCM.
bool write_wav(const string& filename, const vector<double>& samples, int sample_rate) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error opening file for writing: " << filename << "\n";
        return false;
    }
    int16_t bitsPerSample = 16;
    int16_t numChannels = 1;
    int sampleCount = samples.size();
    int byteRate = sample_rate * numChannels * bitsPerSample / 8;
    int16_t blockAlign = numChannels * bitsPerSample / 8;
    int subchunk2Size = sampleCount * numChannels * bitsPerSample / 8;
    int chunkSize = 4 + (8 + 16) + (8 + subchunk2Size);

    // Write WAV header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&chunkSize), 4);
    file.write("WAVE", 4);

    // Write fmt subchunk
    file.write("fmt ", 4);
    uint32_t subchunk1Size = 16;
    file.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    uint16_t audioFormat = 1;
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&numChannels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    // Write data subchunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&subchunk2Size), 4);

    // Find peak value to scale if necessary.
    double max_amp = 0.0;
    for (double s : samples) {
        if (fabs(s) > max_amp) max_amp = fabs(s);
    }
    double scale = (max_amp > 1.0) ? (1.0 / max_amp) : 1.0;

    // Write samples as int16_t
    for (double s : samples) {
        int16_t sample = static_cast<int16_t>(std::max(-1.0, std::min(1.0, s * scale)) * 32767);
        file.write(reinterpret_cast<const char*>(&sample), sizeof(int16_t));
    }
    return true;
}

// --- DFT computation for forward analysis ---
// Computes DFT for k=0 to N/2 and returns amplitude and phase.
// samples: time domain signal, sample_rate: sample rate, returns vector of {amplitude, phase} for each bin.
void compute_dft(const vector<double>& samples, int sample_rate, vector<pair<double, double>>& dft_result) {
    int N = samples.size();
    int n_bins = N / 2 + 1;
    dft_result.resize(n_bins);
    for (int k = 0; k < n_bins; k += FREQUENCY_STEP) { // Increment by FREQUENCY_STEP for speed
        double real = 0.0;
        double imag = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = two_pi * k * n / N;
            real += samples[n] * cos(angle);
            imag -= samples[n] * sin(angle);
        }
        double mag = sqrt(real * real + imag * imag);
        double phase = atan2(imag, real);
        dft_result[k] = make_pair(mag / N, phase);
    }
}

// --- Helper function to get base name (remove extension) ---
string get_basename(const string& filename) {
    size_t last_slash = filename.find_last_of("/\\");
    size_t start = (last_slash == string::npos) ? 0 : last_slash + 1;
    size_t last_dot = filename.find_last_of('.');
    if (last_dot == string::npos || last_dot < start) {
        return filename;
    }
    return filename.substr(0, last_dot);
}

// --- Main program ---
// Usage for forward analysis (DFT):
//   ./a.out -forward <input_audio_file> <output_analysis_file>
// Usage for additive synthesis (reconstruction):
//   ./a.out -reverse <input_analysis_file> <output_audio_file>
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage:\n";
        cerr << "  Forward analysis: " << argv[0] << " -forward <input_audio_file> <output_analysis_file>\n";
        cerr << "  Reverse synthesis: " << argv[0] << " -reverse <input_analysis_file> <output_audio_file>\n";
        return 1;
    }

    string mode = argv[1];
    if (mode == "-forward") {
        string input_file = argv[2];
        string analysis_file = argv[3];
        
        // Convert the input audio file to a 44100 Hz mono WAV using ffmpeg.
        // Make sure ffmpeg is installed and in your PATH.
        string wav_file = "temp.wav";
        string command = "ffmpeg -y -i \"" + input_file + "\" -ac 1 -ar 44100 -f wav " + wav_file;
        cout << "Converting " << input_file << " to WAV format...\n";
        int ret = system(command.c_str());
        if (ret != 0) {
            cerr << "Error converting file with ffmpeg.\n";
            return 1;
        }
        
        // Read the WAV file.
        vector<double> samples;
        int sample_rate;
        if (!read_wav(wav_file, samples, sample_rate)) {
            cerr << "Error reading WAV file.\n";
            return 1;
        }
        cout << "Read " << samples.size() << " samples at " << sample_rate << " Hz.\n";
        
        // Optionally remove the temporary WAV file.
        remove(wav_file.c_str());
        
        // Compute full Fourier analysis (DFT) for bins from 0 to N/2.
        vector<pair<double, double>> dft_result; // pair: amplitude and phase
        compute_dft(samples, sample_rate, dft_result);
        
        // Store the analysis in FourierSound, keeping only the highest amplitude frequencies.
        FourierSound fs;
        fs.sample_rate = sample_rate;
        fs.N = samples.size();
        vector<int> candidate_bins;
        for (int k = 0; k < dft_result.size(); k += FREQUENCY_STEP) {
            candidate_bins.push_back(k);
        }
        sort(candidate_bins.begin(), candidate_bins.end(), [&](int a, int b) {
            return dft_result[a].first > dft_result[b].first;
        });
        int keep_count = max(1, static_cast<int>(candidate_bins.size() * FREQUENCY_KEEP_RATIO));
        vector<int> kept_bins(candidate_bins.begin(), candidate_bins.begin() + keep_count);
        sort(kept_bins.begin(), kept_bins.end());
        for (int k : kept_bins) {
            double freq = k * (static_cast<double>(sample_rate) / fs.N);
            fs.add_sine_wave(freq, dft_result[k].first, dft_result[k].second);
        }
        
        // Write the output analysis file.
        try {
            fs.save(analysis_file);
        } catch (const exception& e) {
            cerr << "Error saving analysis file: " << e.what() << "\n";
            return 1;
        }
        
        cout << "Fourier analysis complete. Analysis saved to " << analysis_file << "\n";
    }
    else if (mode == "-reverse") {
        string analysis_file = argv[2];
        string output_wav = argv[3];
        
        // Read the analysis file.
        ifstream ifs(analysis_file);
        if (!ifs) {
            cerr << "Error opening analysis file: " << analysis_file << "\n";
            return 1;
        }
        int sample_rate, num_sines;
        ifs >> sample_rate >> num_sines;
        vector<SineWave> bins;
        for (int i = 0; i < num_sines; i++) {
            double frequency, amplitude, phase;
            ifs >> frequency >> amplitude >> phase;
            phase = rand();
            bins.push_back({ frequency, amplitude, phase });
        }
        ifs.close();
        
        // Generate one second of audio regardless of the original sample count.
        int N = sample_rate * 2;
        vector<double> synthesis(N, 0.0);
        for (int n = 0; n < N; n++) {
            double sample = 0.0;
            for (size_t i = 0; i < bins.size(); i++) {
                double angle = two_pi * bins[i].frequency * n / sample_rate;
                sample += bins[i].amplitude * cos(angle + bins[i].phase);
            }
            synthesis[n] = sample;
        }
        
        // Write the synthesized audio to a WAV file.
        if (!write_wav(output_wav, synthesis, sample_rate)) {
            cerr << "Error writing output WAV file.\n";
            return 1;
        }
        
        cout << "Additive synthesis complete. Output saved to " << output_wav << "\n";
    }
    else {
        cerr << "Unknown mode. Use -forward or -reverse.\n";
        return 1;
    }
    
    return 0;
}
