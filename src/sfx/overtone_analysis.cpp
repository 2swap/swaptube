#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
extern "C" {
#include <fftw3.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

#define SAMPLE_RATE 44100  // Assumed sample rate (adjust as needed)

// Apply Hann window to the samples
void apply_hann_window(std::vector<double>& samples) {
    int N = samples.size();
    for (int i = 0; i < N; ++i) {
        double hann = 0.5 * (1 - cos((2 * M_PI * i) / (N - 1)));
        samples[i] *= hann;
    }
}

// Perform FFT and extract frequency amplitudes
std::vector<std::pair<double, double>> analyze_frequencies(std::vector<double>& samples) {
    int N = samples.size();
    fftw_complex *out;
    fftw_plan plan;
    std::vector<std::pair<double, double>> frequencies;

    // Apply Hann window
    apply_hann_window(samples);

    // Allocate FFTW memory
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Copy input data
    for (int i = 0; i < N; ++i) {
        in[i][0] = samples[i]; // Real part
        in[i][1] = 0.0; // Imaginary part
    }

    // Perform FFT
    plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Identify frequency peaks
    double bin_width = SAMPLE_RATE / (double) N;
    for (int i = 1; i < N / 2; ++i) {
        double freq = i * bin_width;
        double amplitude = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        frequencies.emplace_back(freq, amplitude);
    }

    // Normalize amplitudes relative to the fundamental
    if (!frequencies.empty()) {
        double fundamental_amplitude = frequencies[0].second;
        for (auto& f : frequencies) {
            f.second /= fundamental_amplitude;
        }
    }

    // Cleanup FFTW
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return frequencies;
}

// Function to decode audio using FFmpeg
std::vector<double> decode_audio(const char* filename) {
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    SwrContext* swrCtx = nullptr;
    int audioStreamIndex = -1;

    std::vector<double> samples;

    // Initialize FFmpeg
    avformat_open_input(&formatCtx, filename, nullptr, nullptr);
    avformat_find_stream_info(formatCtx, nullptr);

    // Find audio stream
    for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            break;
        }
    }
    if (audioStreamIndex == -1) {
        std::cerr << "No audio stream found\n";
        return samples;
    }

    // Open codec
    const AVCodec* codec = avcodec_find_decoder(formatCtx->streams[audioStreamIndex]->codecpar->codec_id);
    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, formatCtx->streams[audioStreamIndex]->codecpar);
    avcodec_open2(codecCtx, codec, nullptr);

    // Resampling context
    swrCtx = swr_alloc();
#if LIBAVUTIL_VERSION_MAJOR >= 57
    av_opt_set_chlayout(swrCtx, "in_chlayout", &codecCtx->ch_layout, 0);
    AVChannelLayout mono_layout;
    av_channel_layout_default(&mono_layout, 1);
    av_opt_set_chlayout(swrCtx, "out_chlayout", &mono_layout, 0);
#else
    av_opt_set_int(swrCtx, "in_channel_layout", codecCtx->channel_layout, 0);
    av_opt_set_int(swrCtx, "out_channel_layout", AV_CH_LAYOUT_MONO, 0);
#endif
    av_opt_set_int(swrCtx, "in_sample_rate", codecCtx->sample_rate, 0);
    av_opt_set_int(swrCtx, "out_sample_rate", SAMPLE_RATE, 0);
    av_opt_set_sample_fmt(swrCtx, "in_sample_fmt", codecCtx->sample_fmt, 0);
    av_opt_set_sample_fmt(swrCtx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);
    swr_init(swrCtx);

    // Read frames
    while (av_read_frame(formatCtx, packet) >= 0) {
        if (packet->stream_index == audioStreamIndex) {
            avcodec_send_packet(codecCtx, packet);
            while (avcodec_receive_frame(codecCtx, frame) == 0) {
                float* buffer = new float[frame->nb_samples];
                swr_convert(swrCtx, (uint8_t**)&buffer, frame->nb_samples, (const uint8_t**)frame->data, frame->nb_samples);
                for (int i = 0; i < frame->nb_samples; i++) {
                    samples.push_back(buffer[i]);
                }
                delete[] buffer;
            }
        }
        av_packet_unref(packet);
    }

    // Cleanup
    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swrCtx);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);

    return samples;
}

// Function to generate output filename with .fft extension
std::string generate_output_filename(const std::string& input_filename) {
    size_t dot_pos = input_filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        return input_filename.substr(0, dot_pos) + ".fft";
    } else {
        return input_filename + ".fft";
    }
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio_file>\n";
        return 1;
    }

    const char* filename = argv[1];
    std::string output_filename = generate_output_filename(filename);

    std::vector<double> samples = decode_audio(filename);
    if (samples.empty()) {
        std::cerr << "Failed to extract samples from audio\n";
        return 1;
    }

    std::vector<std::pair<double, double>> frequencies = analyze_frequencies(samples);

    // Open output file
    std::ofstream outFile(output_filename);
    if (!outFile) {
        std::cerr << "Error opening output file: " << output_filename << "\n";
        return 1;
    }

    // Print total count as first line
    int num_overtones = 10;
    outFile << num_overtones << "\n";

    // Write frequency and relative amplitude to file
    for (const auto& f : frequencies) {
        outFile << f.first << " " << f.second << "\n";
    }

    std::cout << "FFT analysis saved to: " << output_filename << "\n";
    return 0;
}

