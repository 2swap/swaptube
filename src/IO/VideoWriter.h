#pragma once

#include <string>
#include "../Core/Pixels.h"
#include "IoHelpers.h"
#include "LivePlayer.h"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/opt.h>
    #include <libavutil/hwcontext_drm.h>
}

class VideoWriter {
private:
    // note that the FormatContext has shared ownership with audioWriter and MovieWriter
    AVFormatContext *fc = nullptr;

    AVStream *videoStream = nullptr;
    AVCodecContext *videoCodecContext = nullptr;
    AVFrame *yuvpic = nullptr;
    AVPacket pkt = {0};
    unsigned outframe = 0;
    SwsContext* sws_ctx = nullptr;

    bool encode_and_write_frame(AVFrame* frame);
    LivePlayer* lp;

public:
    VideoWriter(AVFormatContext *fc_, const std::string& video_path, int video_width_pixels, int video_height_pixels, int video_framerate_fps);
    void add_frame(uint32_t* device_argb_pixels);
    ~VideoWriter();
};
