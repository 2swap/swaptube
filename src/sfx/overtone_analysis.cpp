#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstring>

using namespace std;

const double two_pi = 2.0 * M_PI;

// Dummy PATH_MANAGER (set directory to current directory)
struct PATH_MANAGER {
    static const string sfx_dir;
};
const string PATH_MANAGER::sfx_dir = "";

// Struct to hold a single sine wave's parameters
struct SineWave {
    double frequency; // in Hz
    double amplitude;
};

// FourierSound class that stores multiple sine waves and writes them in the expected format.
class FourierSound {
public:
    vector<SineWave> sine_waves;

    // Add a sine wave to the FourierSound object.
    void add_sine_wave(double frequency, double amplitude) {
        sine_waves.push_back({frequency, amplitude});
    }

    // Save to file.
    // File format:
    //   <number_of_sine_waves>
    //   <frequency> <amplitude>
    //   ...
    bool save(const string& filename) const {
        ofstream ofs(PATH_MANAGER::sfx_dir + filename);
        if (!ofs) {
            throw runtime_error("Error opening file for writing: " + filename);
        }
        ofs << sine_waves.size() << "\n";
        for (const auto& sine : sine_waves) {
            ofs << sine.frequency << " " << sine.amplitude << "\n";
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

// --- Goertzel algorithm ---
// Computes the amplitude of the frequency component (target_freq) in the samples.
double goertzel(const vector<double>& samples, double target_freq, int sample_rate) {
    int N = samples.size();
    double omega = 2.0 * M_PI * target_freq / sample_rate;
    double coeff = 2.0 * cos(omega);
    double s_prev = 0.0;
    double s_prev2 = 0.0;
    for (int i = 0; i < N; i++) {
        double s = samples[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    double real = s_prev - s_prev2 * cos(omega);
    double imag = s_prev2 * sin(omega);
    double magnitude = sqrt(real * real + imag * imag);
    // Normalize by the number of samples.
    return magnitude / N;
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
// Usage: ./a.out input_audio_file fundamental_frequency
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_audio_file> <fundamental_frequency>\n";
        return 1;
    }
    
    string input_file = argv[1];
    double fundamental;
    try {
        fundamental = stod(argv[2]);
    } catch (const exception& e) {
        cerr << "Invalid fundamental frequency: " << argv[2] << "\n";
        return 1;
    }
    
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
    
    // Perform harmonic analysis using the Goertzel algorithm.
    // We compute each harmonic (n * fundamental) until we exceed the Nyquist frequency.
    FourierSound fs;
    int harmonic_index = 1;
    while (true) {
        double freq = harmonic_index * fundamental;
        if (freq > sample_rate / 2) break; // do not exceed the Nyquist limit
        double amplitude = goertzel(samples, freq, sample_rate);
        fs.add_sine_wave(freq, amplitude);
        harmonic_index++;
    }
    
    // Write the output file in the expected FFT format.
    string base = get_basename(input_file);
    string output_file = base + ".fft";
    try {
        fs.save(output_file);
    } catch (const exception& e) {
        cerr << "Error saving FFT file: " << e.what() << "\n";
        return 1;
    }
    
    cout << "FFT analysis complete. Output saved to " << output_file << "\n";
    return 0;
}

