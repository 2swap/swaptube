#pragma once

#include <string>
#include <cstdint>

extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct SwsContext;
struct AVPacket;
struct AVFrame;
}

#include "../Core/Pixels.h"

class MP4FrameReader {
public:
    explicit MP4FrameReader(const std::string &vn);
    ~MP4FrameReader();

    void change_video(const std::string &vn);
    // get_frame returns true on EOF / failure to find frame, false on success.
    bool get_frame(int frame_index, int target_width, int target_height, Pixels& pix);

private:
    std::string video_name;
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

    void open_file();
    void ensure_scaler(int width, int height, int &scaled_width, int &scaled_height);
    void reset();
    void cleanup();
};
