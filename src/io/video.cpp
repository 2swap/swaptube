#pragma once
#include <iostream>
#include <string>
#include <regex>
#include <cassert>
#include "DebugPlot.h"
extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

using namespace std;

static DebugPlot ffmpeg_output_plot("ffmpeg output", vector<string>{"frame", "qp", "nal", "poc", "i", "p", "skip", "size"});

// Function to redirect stderr to a pipe
int redirect_stderr(int pipefd[2]) {
    fflush(stderr);
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return -1;
    }

    // Save the original stderr
    int original_stderr = dup(STDERR_FILENO);

    // Redirect stderr to the write end of the pipe
    if (dup2(pipefd[1], STDERR_FILENO) == -1) {
        perror("dup2");
        return -1;
    }

    // Close the write end of the pipe in the parent process
    close(pipefd[1]);

    return original_stderr;
}

// Function to restore the original stderr
void restore_stderr(int original_stderr) {
    fflush(stderr);
    dup2(original_stderr, STDERR_FILENO);
    close(original_stderr);
}

// Function to read from a file descriptor into a string
string read_from_fd(int fd) {
    string output;
    char buffer[1024];
    ssize_t bytes_read;

    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        output.append(buffer, bytes_read);
    }

    return output;
}

// Function to parse the debug output
void parse_debug_output(const string& output) {
    istringstream iss(output);
    string line;
    regex regex_pattern(R"(frame=\s*(\d+)\s*QP=(\d+\.\d+)\s+NAL=(\d+)\s+Slice:([IPBS])\s+Poc:(\d+)\s+I:(\d+)\s+P:(\d+)\s+SKIP:(\d+)\s+size=(\d+)\s+bytes)");
    regex regex_pattern_empty_line(R"(\s*)");
    while (getline(iss, line)) {
        smatch match;

        if (regex_search(line, match, regex_pattern) && match.size() == 10) {
            double frame = stoi(match[1].str());
            double qp    = stof(match[2].str());
            double nal   = stoi(match[3].str());
            string slice =      match[4].str();
            double poc   = stoi(match[5].str());
            double i     = stoi(match[6].str());
            double p     = stoi(match[7].str());
            double skip  = stoi(match[8].str());
            double size  = stoi(match[9].str());
            ffmpeg_output_plot.add_datapoint(vector<double>{frame, qp, nal, poc, i, p, skip, size});
        } else if(regex_search(line, match, regex_pattern_empty_line)) {
            // do nothing, i guess?
        } else {
            // If the string did not match the expected format, dump it to stderr
            cerr << "Failed to parse cerr output from encoder: " << line << endl;
        }
    }
}

/*
 * Wrapper around avcodec_send_frame which captures its output to stderr.
 */
int send_frame(AVCodecContext* vcc, AVFrame* frame){
    int pipefd[2];
    int original_stderr = redirect_stderr(pipefd);
    int ret = avcodec_send_frame(vcc, frame);
    restore_stderr(original_stderr);
    string debug_output = read_from_fd(pipefd[0]);
    close(pipefd[0]);
    parse_debug_output(debug_output);
    return ret;
}

class VideoWriter {
private:

    //note that the FormatContext has shared ownership with audioWriter and MovieWriter
    AVFormatContext *fc = nullptr;

    AVStream *videoStream = nullptr;
    AVCodecContext *videoCodecContext = nullptr;
    AVFrame *yuvpic = nullptr;
    AVPacket pkt = {0};
    unsigned outframe = 0;

    bool encode_and_write_frame(AVFrame* frame){
        if(frame != NULL){
            int ret = send_frame(videoCodecContext, frame);
            if (ret<0) {
                cout << "Failed encoding video!" << endl;
                exit(1);
            }
        }

        int ret2 = avcodec_receive_packet(videoCodecContext, &pkt);
        if (ret2 == AVERROR(EAGAIN) || ret2 == AVERROR_EOF) {
            return false;
        } else if (ret2!=0) {
            cout << "Failed to receive video packet!" << endl;
            exit(1);
        }

        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, { 1, VIDEO_FRAMERATE }, videoStream->time_base);

        pkt.stream_index = videoStream->index;

        // Write the encoded frame to the mp4 file.
        int pipefd[2];
        int original_stderr = redirect_stderr(pipefd);
        av_interleaved_write_frame(fc, &pkt);
        restore_stderr(original_stderr);
        close(pipefd[0]);

        return true;
    }

public:
    VideoWriter(AVFormatContext *fc_) : fc(fc_) {
        av_log_set_level(AV_LOG_DEBUG);

        // Setting up the codec.
        const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
        AVDictionary* opt = NULL;
        av_dict_set(&opt, "preset", "ultrafast", 0);
        av_dict_set(&opt, "crf", "18", 0);

        videoStream = avformat_new_stream(fc, codec);
        if (!videoStream) {
            cout << "Failed to create new videostream!" << endl;
            exit(1);
        }

        videoCodecContext = avcodec_alloc_context3(codec);
        videoCodecContext->width = VIDEO_WIDTH;
        videoCodecContext->height = VIDEO_HEIGHT;
        videoCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
        videoCodecContext->time_base = { 1, VIDEO_FRAMERATE };

        avcodec_parameters_from_context(videoStream->codecpar, videoCodecContext);

        avcodec_open2(videoCodecContext, codec, &opt);
        av_dict_free(&opt);

        videoStream->time_base = { 1, VIDEO_FRAMERATE };
        av_dump_format(fc, 0, PATH_MANAGER.video_output.c_str(), 1);
        avio_open(&fc->pb, PATH_MANAGER.video_output.c_str(), AVIO_FLAG_WRITE);
        int ret = avformat_write_header(fc, &opt);
        if (ret < 0) {
            cout << "Failed to write header!" << endl;
            exit(1);
        }
        av_dict_free(&opt);

        // Allocating memory for each RGB and YUV frame.
        yuvpic = av_frame_alloc();
        yuvpic->format = AV_PIX_FMT_YUV420P;
        yuvpic->width = VIDEO_WIDTH;
        yuvpic->height = VIDEO_HEIGHT;
        av_frame_get_buffer(yuvpic, 1);
    }

    void add_frame(const Pixels& p) {
        if (p.w != VIDEO_WIDTH || p.h != VIDEO_HEIGHT)
            throw runtime_error("Frame dimensions were expected to be (" + to_string(VIDEO_WIDTH) + ", " + to_string(VIDEO_HEIGHT) + "), but they were instead (" + to_string(p.w) + ", " + to_string(p.h) + ")!");

        uint8_t* y_plane = yuvpic->data[0];
        uint8_t* u_plane = yuvpic->data[1];
        uint8_t* v_plane = yuvpic->data[2];

        int y_stride = yuvpic->linesize[0];
        int u_stride = yuvpic->linesize[1];
        int v_stride = yuvpic->linesize[2];

        for (unsigned int y = 0; y < VIDEO_HEIGHT; y++) {
            for (unsigned int x = 0; x < VIDEO_WIDTH; x++) {
                int a, r, g, b;
                p.get_pixel_by_channels(x, y, a, r, g, b);

                double alpha = a / 255.0;
                double one_minus_alpha = 1.0 - alpha;
                const int background_r = 0;
                const int background_g = 0;
                const int background_b = 0;
                r = r * alpha + background_r * one_minus_alpha;
                g = g * alpha + background_g * one_minus_alpha;
                b = b * alpha + background_b * one_minus_alpha;

                // Convert RGB to YUV
                double y_value = 0.299 * r + 0.587 * g + 0.114 * b;

                // Assign to Y plane
                y_plane[y * y_stride + x] = static_cast<uint8_t>(std::clamp(y_value, 0.0, 255.0));

                // Assign to U and V planes (downsample 2x2)
                if (x % 2 == 0 && y % 2 == 0) {
                    double u_value = -0.14713 * r - 0.28886 * g + 0.436 * b + 128;
                    double v_value = 0.615 * r - 0.51499 * g - 0.10001 * b + 128;
                    int u_idx = (y / 2) * u_stride + (x / 2);
                    int v_idx = (y / 2) * v_stride + (x / 2);
                    u_plane[u_idx] = static_cast<uint8_t>(std::clamp(u_value, 0.0, 255.0));
                    v_plane[v_idx] = static_cast<uint8_t>(std::clamp(v_value, 0.0, 255.0));
                }
            }
        }

        // The PTS of the frame are just in a reference unit,
        // unrelated to the format we are using. We set them,
        // for instance, as the corresponding frame number.
        yuvpic->pts = outframe;
        outframe++;

        encode_and_write_frame(yuvpic);
    }
    void cleanup() {
        // Writing the delayed video frames
        while(encode_and_write_frame(NULL));

        // Closing the video codec.
        avcodec_close(videoCodecContext);

        // Freeing video specific resources
        av_frame_free(&yuvpic);

        av_packet_unref(&pkt);

        av_write_trailer(fc);
        avio_closep(&fc->pb);
        avformat_free_context(fc);
    }
};
