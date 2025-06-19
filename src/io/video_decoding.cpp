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

#include "../misc/pixels.h"

// mp4_num_frames:
// Returns the number of frames in the given MP4 video file.
/*int mp4_num_frames(const std::string &filename) {
    std::string fname = filename;
    if (fname.length() < 4 || fname.substr(fname.length() - 4) != ".mp4") {
        fname += ".mp4";
    }
    fname = PATH_MANAGER.this_project_media_dir + fname;

    AVFormatContext* fmtCtx = nullptr;
    if (avformat_open_input(&fmtCtx, fname.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Could not open video file: " + fname);
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Could not retrieve stream info from file: " + fname);
    }

    // Find the first video stream
    int videoStreamIdx = -1;
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIdx = i;
            break;
        }
    }
    if (videoStreamIdx == -1) {
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("No video stream found in file: " + fname);
    }

    AVStream* videoStream = fmtCtx->streams[videoStreamIdx];

    int64_t nb_frames = videoStream->nb_frames;
    if (nb_frames == 0) {
        // nb_frames may be zero for some containers, fallback to duration / frame time base estimate
        AVRational frame_rate = av_guess_frame_rate(fmtCtx, videoStream, nullptr);
        if (frame_rate.num == 0 || frame_rate.den == 0) {
            avformat_close_input(&fmtCtx);
            throw std::runtime_error("Could not determine frame rate for file: " + fname);
        }
        int64_t duration = videoStream->duration;
        if (duration == AV_NOPTS_VALUE) {
            duration = fmtCtx->duration / AV_TIME_BASE;
            if (duration <= 0) {
                avformat_close_input(&fmtCtx);
                throw std::runtime_error("Could not determine duration for file: " + fname);
            }
        }
        // Calculate approximate frame count
        nb_frames = (int64_t)((double)duration * frame_rate.num / frame_rate.den);
    }

    avformat_close_input(&fmtCtx);
    return static_cast<int>(nb_frames);
}*/

// mp4_to_pix_bounding_box:
// Loads the specified frame (by index) from the given MP4 file,
// scales it to (target_width x target_height) using a bounding box approach,
// and returns the result as a Pixels object.
Pixels mp4_to_pix_bounding_box(const std::string &video_name, int target_width, int target_height, int frame_index, bool& no_more_frames) {
    no_more_frames = false;
    string filename = video_name;
    if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".mp4") {
        filename += ".mp4";
    }
    filename = PATH_MANAGER.this_project_media_dir + filename; 
    // Open the input file and read header
    AVFormatContext* fmtCtx = nullptr;
    if (avformat_open_input(&fmtCtx, filename.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Could not open video file: " + filename);
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Could not retrieve stream info from file: " + filename);
    }

    // Find the first video stream
    int videoStreamIdx = -1;
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIdx = i;
            break;
        }
    }
    if (videoStreamIdx == -1) {
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("No video stream found in file: " + filename);
    }

    AVCodecParameters* codecPar = fmtCtx->streams[videoStreamIdx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecPar->codec_id);
    if (!codec) {
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Unsupported codec in file: " + filename);
    }

    // Allocate and open codec context
    AVCodecContext* codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Failed to allocate codec context.");
    }
    if (avcodec_parameters_to_context(codecCtx, codecPar) < 0) {
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Couldn't copy codec parameters to context.");
    }
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Couldn't open codec.");
    }

    // Allocate packet and frames
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* frameRGBA = av_frame_alloc();
    if (!packet || !frame || !frameRGBA) {
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&frameRGBA);
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Failed to allocate packet or frames.");
    }

    // Allocate buffer for the converted RGBA frame
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGBA, target_width, target_height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    if (!buffer) {
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&frameRGBA);
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Could not allocate buffer for image conversion.");
    }
    av_image_fill_arrays(frameRGBA->data, frameRGBA->linesize, buffer, AV_PIX_FMT_RGBA, target_width, target_height, 1);

    // Set up the conversion context to convert from the native pixel format to RGBA.
    struct SwsContext* swsCtx = sws_getContext(
        codecCtx->width,
        codecCtx->height,
        codecCtx->pix_fmt,
        target_width,
        target_height,
        AV_PIX_FMT_RGBA,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr
    );
    if (!swsCtx) {
        av_free(buffer);
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&frameRGBA);
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        throw std::runtime_error("Could not initialize the conversion context.");
    }

    // Decode packets until we reach the desired frame index.
    int currentFrame = 0;
    bool foundFrame = false;
    while (av_read_frame(fmtCtx, packet) >= 0) {
        if (packet->stream_index == videoStreamIdx) {
            int sendResult = avcodec_send_packet(codecCtx, packet);
            if (sendResult < 0) {
                av_packet_unref(packet);
                continue;
            }
            while (true) {
                int receiveResult = avcodec_receive_frame(codecCtx, frame);
                if (receiveResult == AVERROR(EAGAIN) || receiveResult == AVERROR_EOF)
                    break;
                else if (receiveResult < 0) {
                    sws_freeContext(swsCtx);
                    av_free(buffer);
                    av_packet_free(&packet);
                    av_frame_free(&frame);
                    av_frame_free(&frameRGBA);
                    avcodec_free_context(&codecCtx);
                    avformat_close_input(&fmtCtx);
                    throw std::runtime_error("Error during decoding.");
                }
                // Check if this is the frame we want.
                if (currentFrame == frame_index) {
                    // Convert the frame to RGBA with the target dimensions.
                    sws_scale(swsCtx,
                              frame->data,
                              frame->linesize,
                              0,
                              codecCtx->height,
                              frameRGBA->data,
                              frameRGBA->linesize);
                    foundFrame = true;
                    break;
                }
                currentFrame++;
            }
        }
        av_packet_unref(packet);
        if (foundFrame)
            break;
    }

    if (!foundFrame) {
        sws_freeContext(swsCtx);
        av_free(buffer);
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&frameRGBA);
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        no_more_frames = true;
        return Pixels();
        //throw std::runtime_error("Could not decode requested frame (frame index: " + std::to_string(frame_index) + ").");
    }

    // Create a Pixels object and copy the RGBA data into it.
    Pixels pix(target_width, target_height);
    // Assume frameRGBA->data[0] contains the RGBA data.
    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            int offset = y * frameRGBA->linesize[0] + x * 4;
            uint8_t r = frameRGBA->data[0][offset];
            uint8_t g = frameRGBA->data[0][offset + 1];
            uint8_t b = frameRGBA->data[0][offset + 2];
            uint8_t a = frameRGBA->data[0][offset + 3];
            // Combine the channels into a single integer using your helper function.
            int color = argb(a, r, g, b);
            pix.set_pixel(x, y, color);
        }
    }

    // Clean up allocated resources.
    sws_freeContext(swsCtx);
    av_free(buffer);
    av_packet_free(&packet);
    av_frame_free(&frame);
    av_frame_free(&frameRGBA);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&fmtCtx);

    return pix;
}

