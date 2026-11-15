#pragma once

#include <string>
#include "VideoWriter.h"
#include "../Core/Pixels.h"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
}

// Software (libx264) encoder for machines without a hardware video encoder.
// Selected with go.sh -e.
// Output is 8-bit H.264, so colors are quantized slightly differently than the
// 10-bit hardware path.
class CpuVideoWriter : public VideoWriter {
private:
    // note that the FormatContext has shared ownership with AudioWriter
    AVFormatContext *fc = nullptr;

    AVStream *videoStream = nullptr;
    AVCodecContext *videoCodecContext = nullptr;
    AVFrame *yuvpic = nullptr;
    AVPacket *pkt = nullptr;
    SwsContext *sws_ctx = nullptr;
    Pixels host_frame;
    unsigned outframe = 0;

    bool encode_and_write_frame(AVFrame* frame);

public:
    CpuVideoWriter(AVFormatContext *fc_, const std::string& video_path, int video_width_pixels, int video_height_pixels, int video_framerate_fps);
    void add_frame(uint32_t* device_argb_pixels) override;
    ~CpuVideoWriter() override;
};
