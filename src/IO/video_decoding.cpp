#pragma once
#include <stdexcept>
#include <string>
#include <cstdint>
#include <iostream>
#include "IoHelpers.cpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include "../misc/Pixels.h"

// mp4_to_pix_bounding_box:
// Loads the specified frame (by index) from the given MP4 file,
// scales it to (target_width x target_height) using a bounding box approach,
// and returns the result as a Pixels object.

class MP4FrameReader {
public:
    explicit MP4FrameReader(const std::string &video_name)
        : fmtCtx(nullptr), codecCtx(nullptr), swsCtx(nullptr),
          packet(nullptr), frame(nullptr), frameRGBA(nullptr),
          buffer(nullptr), buffer_size(0), currentFrame(-1), videoStreamIdx(-1)
    {
        open_file(video_name);
    }

    ~MP4FrameReader() {
        cleanup();
    }

    void change_video(const std::string &video_name) {
        reset();
        open_file(video_name);
    }

    Pixels get_frame(int frame_index, int target_width, int target_height, bool &no_more_frames) {
        no_more_frames = false;

        if (frame_index <= currentFrame) {
            // Requested an earlier frame -> reset decoding.
            reset();
            open_file(filename);
        }

        // Ensure scaling context matches requested output size.
        ensure_scaler(target_width, target_height);

        // Decode until reaching the desired frame.
        while (av_read_frame(fmtCtx, packet) >= 0) {
            if (packet->stream_index == videoStreamIdx) {
                if (avcodec_send_packet(codecCtx, packet) < 0) {
                    av_packet_unref(packet);
                    continue;
                }

                while (true) {
                    int receiveResult = avcodec_receive_frame(codecCtx, frame);
                    if (receiveResult == AVERROR(EAGAIN) || receiveResult == AVERROR_EOF)
                        break;
                    else if (receiveResult < 0)
                        throw std::runtime_error("Error during decoding.");

                    currentFrame++;
                    if (currentFrame == frame_index) {
                        // Convert frame to RGBA
                        sws_scale(swsCtx,
                                  frame->data,
                                  frame->linesize,
                                  0,
                                  codecCtx->height,
                                  frameRGBA->data,
                                  frameRGBA->linesize);

                        // Copy into Pixels
                        Pixels pix(target_width, target_height);
                        for (int y = 0; y < target_height; y++) {
                            for (int x = 0; x < target_width; x++) {
                                int offset = y * frameRGBA->linesize[0] + x * 4;
                                uint8_t r = frameRGBA->data[0][offset];
                                uint8_t g = frameRGBA->data[0][offset + 1];
                                uint8_t b = frameRGBA->data[0][offset + 2];
                                uint8_t a = frameRGBA->data[0][offset + 3];
                                pix.set_pixel(x, y, argb(a, r, g, b));
                            }
                        }

                        av_packet_unref(packet);
                        return pix;
                    }
                }
            }
            av_packet_unref(packet);
        }

        // Reached EOF without finding the frame.
        no_more_frames = true;
        return Pixels();
    }

private:
    std::string filename;
    AVFormatContext *fmtCtx;
    AVCodecContext *codecCtx;
    SwsContext *swsCtx;
    AVPacket *packet;
    AVFrame *frame;
    AVFrame *frameRGBA;
    uint8_t *buffer;
    int buffer_size;
    int currentFrame;
    int videoStreamIdx;

    void open_file(const std::string &video_name) {
        filename = video_name;
        if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".mp4")
            filename += ".mp4";
        filename = PATH_MANAGER.this_project_media_dir + filename;

        if (avformat_open_input(&fmtCtx, filename.c_str(), nullptr, nullptr) != 0)
            throw std::runtime_error("Could not open video file: " + filename);
        if (avformat_find_stream_info(fmtCtx, nullptr) < 0)
            throw std::runtime_error("Could not retrieve stream info from file: " + filename);

        // Find video stream
        for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
            if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStreamIdx = i;
                break;
            }
        }
        if (videoStreamIdx == -1)
            throw std::runtime_error("No video stream found in file: " + filename);

        AVCodecParameters *codecPar = fmtCtx->streams[videoStreamIdx]->codecpar;
        const AVCodec *codec = avcodec_find_decoder(codecPar->codec_id);
        if (!codec)
            throw std::runtime_error("Unsupported codec in file: " + filename);

        codecCtx = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codecCtx, codecPar) < 0)
            throw std::runtime_error("Couldn't copy codec parameters.");
        if (avcodec_open2(codecCtx, codec, nullptr) < 0)
            throw std::runtime_error("Couldn't open codec.");

        packet = av_packet_alloc();
        frame = av_frame_alloc();
        frameRGBA = av_frame_alloc();

        currentFrame = -1;
    }

    void ensure_scaler(int width, int height) {
        // Recreate SWS context and buffer if size changed
        if (swsCtx && (frameRGBA->width == width && frameRGBA->height == height))
            return;

        if (swsCtx) sws_freeContext(swsCtx);
        if (buffer) av_free(buffer);

        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGBA, width, height, 1);
        buffer = (uint8_t *)av_malloc(numBytes);
        buffer_size = numBytes;
        av_image_fill_arrays(frameRGBA->data, frameRGBA->linesize, buffer,
                             AV_PIX_FMT_RGBA, width, height, 1);
        frameRGBA->width = width;
        frameRGBA->height = height;

        swsCtx = sws_getContext(codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
                                width, height, AV_PIX_FMT_RGBA,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);

        if (!swsCtx)
            throw std::runtime_error("Could not initialize scaling context.");
    }

    void reset() {
        cleanup();
        fmtCtx = nullptr;
        codecCtx = nullptr;
        swsCtx = nullptr;
        packet = nullptr;
        frame = nullptr;
        frameRGBA = nullptr;
        buffer = nullptr;
        buffer_size = 0;
        currentFrame = -1;
        videoStreamIdx = -1;
    }

    void cleanup() {
        if (swsCtx) sws_freeContext(swsCtx);
        if (buffer) av_free(buffer);
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        if (frameRGBA) av_frame_free(&frameRGBA);
        if (codecCtx) avcodec_free_context(&codecCtx);
        if (fmtCtx) avformat_close_input(&fmtCtx);
    }
};

