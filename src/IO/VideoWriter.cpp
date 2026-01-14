#pragma once
#include <iostream>
#include <string>
#include <regex>
#include <cassert>
#include "DebugPlot.h"
#include "../Core/Pixels.h"
#include "IoHelpers.cpp"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/opt.h>
}

extern "C"
void alpha_overlay_cuda(unsigned int* src_host,
                        int width, int height,
                        unsigned int bg_color);

using namespace std;

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
            //ffmpeg_output_plot.add_datapoint(vector<double>{frame, qp, nal, poc, i, p, skip, size});
        } else if(regex_search(line, match, regex_pattern_empty_line)) {
            // do nothing, i guess?
        } else {
            // If the string did not match the expected format, dump it to stderr
            cout << "Failed to parse cerr output from encoder: " << line << endl;
            // This happens in nominal conditions, do not failout!
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
    SwsContext* sws_ctx = nullptr;

    bool encode_and_write_frame(AVFrame* frame){
        int ret = send_frame(videoCodecContext, frame);
        if (ret<0 && frame != NULL) throw runtime_error("Failed to encode video!");

        int ret2 = avcodec_receive_packet(videoCodecContext, &pkt);
        if (ret2 == AVERROR(EAGAIN) || ret2 == AVERROR_EOF) return false;
        else if (ret2!=0) throw runtime_error("Failed to receive video packet!");

        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, { 1, FRAMERATE }, videoStream->time_base);

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
    VideoWriter(AVFormatContext *fc_, const string& video_path) : fc(fc_) {
        av_log_set_level(AV_LOG_DEBUG);

        // Setting up the codec.
        const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) {
            throw runtime_error("Failed to find video codec");
        }

        AVDictionary* opt = NULL;
        av_dict_set(&opt, "preset", "medium", 0);
        av_dict_set(&opt, "tune", "film", 0);
        av_dict_set(&opt, "crf", "18", 0);

        videoStream = avformat_new_stream(fc, codec);
        if (!videoStream) {
            throw runtime_error("Failed to create new videostream!");
        }

        videoCodecContext = avcodec_alloc_context3(codec);
        if (!videoCodecContext) {
            throw runtime_error("Failed to allocate video codec context.");
        }
        videoCodecContext->width = VIDEO_WIDTH;
        videoCodecContext->height = VIDEO_HEIGHT;
        videoCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;//10LE;
        videoCodecContext->time_base = videoStream->time_base = { 1, FRAMERATE };
        videoCodecContext->color_range = AVCOL_RANGE_JPEG;
        videoCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        int ret = avcodec_open2(videoCodecContext, codec, &opt);
        if (ret < 0) {
            throw runtime_error("Failed avcodec_open2!");
        }
        av_dict_free(&opt);

        ret = avcodec_parameters_from_context(videoStream->codecpar, videoCodecContext);
        if (ret < 0) {
            throw runtime_error("Failed avcodec_parameters_from_context!");
        }

        ret = avio_open(&fc->pb, video_path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            throw runtime_error("Failed avio_open!");
        }

        ret = avformat_write_header(fc, &opt);
        if (ret < 0) {
            cout << "Failed to write header: " << ff_errstr(ret) << endl;
            throw runtime_error("Failed to write header!");
        }

        // Allocating memory for each YUV frame.
        yuvpic = av_frame_alloc();
        yuvpic->format = AV_PIX_FMT_YUV420P;
        yuvpic->width = VIDEO_WIDTH;
        yuvpic->height = VIDEO_HEIGHT;
        av_frame_get_buffer(yuvpic, 1);

        sws_ctx = sws_getContext(
            VIDEO_WIDTH, VIDEO_HEIGHT, AV_PIX_FMT_BGRA,
            VIDEO_WIDTH, VIDEO_HEIGHT, AV_PIX_FMT_YUV420P,
            SWS_BICUBIC,
            nullptr, nullptr, nullptr);

        if (!sws_ctx) {
            throw std::runtime_error("Failed to create SwsContext for RGB->YUV conversion!");
        }
    }

    void add_frame(Pixels& p) {
        if (p.w != VIDEO_WIDTH || p.h != VIDEO_HEIGHT)
            throw runtime_error("Frame dimensions were expected to be (" + to_string(VIDEO_WIDTH) + ", " + to_string(VIDEO_HEIGHT) + "), but they were instead (" + to_string(p.w) + ", " + to_string(p.h) + ")!");

        // Allocate a temporary RGB frame wrapper (no copy of pixel data)
        AVFrame* rgb_frame = av_frame_alloc();
        rgb_frame->format = AV_PIX_FMT_BGRA;
        rgb_frame->width  = VIDEO_WIDTH;
        rgb_frame->height = VIDEO_HEIGHT;

        #ifdef USE_CUDA
        alpha_overlay_cuda(reinterpret_cast<unsigned int*>(p.pixels.data()), VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_BACKGROUND_COLOR);
        #endif

        // Point FFmpeg directly to your source data
        rgb_frame->data[0] = reinterpret_cast<uint8_t*>(p.pixels.data());
        rgb_frame->linesize[0] = VIDEO_WIDTH * 4;

        // Convert RGB → YUV using swscale
        sws_scale(sws_ctx,
                  rgb_frame->data,
                  rgb_frame->linesize,
                  0, VIDEO_HEIGHT,
                  yuvpic->data,
                  yuvpic->linesize);

        rgb_frame->data[0] = nullptr;
        av_frame_free(&rgb_frame);

        // Assign PTS and encode
        yuvpic->pts = outframe++;
        encode_and_write_frame(yuvpic);
    }

    ~VideoWriter() {
        cout << "Cleaning up VideoWriter..." << endl;

        // Writing the delayed video frames
        while(encode_and_write_frame(NULL));

        // Free the video codec context.
        avcodec_free_context(&videoCodecContext);

        // Freeing video specific resources
        av_frame_free(&yuvpic);

        av_packet_unref(&pkt);

        av_write_trailer(fc);
        avio_closep(&fc->pb);
        avformat_free_context(fc);

        if (sws_ctx) sws_freeContext(sws_ctx);
    }
};
