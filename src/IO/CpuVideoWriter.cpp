#include "CpuVideoWriter.h"
#include "IoHelpers.h"
#include "Writer.h"

#include <iostream>
#include <stdexcept>

using namespace std;

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

bool CpuVideoWriter::encode_and_write_frame(AVFrame* frame) {
    int ret = avcodec_send_frame(videoCodecContext, frame);
    if (ret < 0) throw runtime_error("avcodec_send_frame failed: " + ff_errstr(ret));

    while (true) {
        int ret2 = avcodec_receive_packet(videoCodecContext, pkt);
        if (ret2 == 0) {
            av_packet_rescale_ts(pkt, videoCodecContext->time_base, videoStream->time_base);
            pkt->stream_index = videoStream->index;
            av_interleaved_write_frame(fc, pkt);
            av_packet_unref(pkt);
        } else if (ret2 == AVERROR(EAGAIN)) {
            return true;
        } else if (ret2 == AVERROR_EOF) {
            cout << "Encoder flushed, no more packets to receive." << endl;
            return false;
        } else
            throw runtime_error("Failed to receive video packet: " + ff_errstr(ret2));
    }
}

CpuVideoWriter::CpuVideoWriter(AVFormatContext *fc_, const string& video_path, int video_width_pixels, int video_height_pixels, int video_framerate_fps) : fc(fc_), host_frame(ivec2(video_width_pixels, video_height_pixels)) {
    av_log_set_level(AV_LOG_ERROR);
    cout << "Using CPU video encoding (libx264, 8-bit). Output colors will differ slightly from the hardware encoder." << endl;

    if (video_width_pixels % 2 != 0 || video_height_pixels % 2 != 0)
        throw runtime_error("CPU encoding (yuv420p) requires even video dimensions, got " + to_string(video_width_pixels) + "x" + to_string(video_height_pixels));

    const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) throw runtime_error("libx264 encoder is not available in this FFmpeg build.");

    videoStream = avformat_new_stream(fc, codec);
    if (!videoStream) throw runtime_error("Failed to create new videostream!");

    videoCodecContext = avcodec_alloc_context3(codec);
    if (!videoCodecContext) throw runtime_error("Failed to allocate video codec context.");
    videoCodecContext->width = video_width_pixels;
    videoCodecContext->height = video_height_pixels;
    videoCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
    // Same BT.709 limited-range tagging as the hardware path
    videoCodecContext->colorspace = AVCOL_SPC_BT709;
    videoCodecContext->color_primaries = AVCOL_PRI_BT709;
    videoCodecContext->color_trc = AVCOL_TRC_BT709;
    videoCodecContext->color_range = AVCOL_RANGE_MPEG;
    videoCodecContext->time_base = { 1, video_framerate_fps };
    videoCodecContext->framerate = { video_framerate_fps, 1 };
    if (fc->oformat->flags & AVFMT_GLOBALHEADER)
        videoCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    AVDictionary* opt = nullptr;
    av_dict_set(&opt, "preset", "veryfast", 0);
    av_dict_set(&opt, "crf", "18", 0);
    int ret = avcodec_open2(videoCodecContext, codec, &opt);
    av_dict_free(&opt);
    if (ret < 0) throw runtime_error("Failed to open libx264: " + ff_errstr(ret));

    ret = avcodec_parameters_from_context(videoStream->codecpar, videoCodecContext);
    if (ret < 0) throw runtime_error("Failed avcodec_parameters_from_context: " + ff_errstr(ret));
    videoStream->time_base = videoCodecContext->time_base;

    ret = avio_open(&fc->pb, video_path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) throw runtime_error("Failed avio_open: " + ff_errstr(ret));

    ret = avformat_write_header(fc, nullptr);
    if (ret < 0) throw runtime_error("Failed to write header: " + ff_errstr(ret));

    yuvpic = av_frame_alloc();
    if (!yuvpic) throw runtime_error("Failed to allocate frame!");
    yuvpic->format = AV_PIX_FMT_YUV420P;
    yuvpic->width = video_width_pixels;
    yuvpic->height = video_height_pixels;
    ret = av_frame_get_buffer(yuvpic, 0);
    if (ret < 0) throw runtime_error("Failed to allocate frame buffer: " + ff_errstr(ret));

    pkt = av_packet_alloc();
    if (!pkt) throw runtime_error("Failed to allocate packet!");

    // Our ARGB words are little-endian, so in memory they are BGRA bytes.
    sws_ctx = sws_getContext(video_width_pixels, video_height_pixels, AV_PIX_FMT_BGRA,
                             video_width_pixels, video_height_pixels, AV_PIX_FMT_YUV420P,
                             SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx) throw runtime_error("Failed to create swscale context!");
    // Full-range RGB in, BT.709 limited range out, matching argb_to_p010
    const int* bt709 = sws_getCoefficients(SWS_CS_ITU709);
    sws_setColorspaceDetails(sws_ctx, bt709, 1, bt709, 0, 0, 1 << 16, 1 << 16);
}

void CpuVideoWriter::add_frame(uint32_t* device_pixels) {
    const ivec2 wh = host_frame.wh;
    cuda_copy_pixels_to_host(host_frame.pixels.data(), wh.x * wh.y, device_pixels);

    // Composite onto the background color, as argb_to_p010 does on the GPU
    const uint32_t bg = get_video_background_color();
    for (unsigned int& px : host_frame.pixels) px = color_combine(bg, px);

    int ret = av_frame_make_writable(yuvpic);
    if (ret < 0) throw runtime_error("Failed to make frame writable: " + ff_errstr(ret));

    const uint8_t* src[1] = { reinterpret_cast<const uint8_t*>(host_frame.pixels.data()) };
    const int src_stride[1] = { 4 * wh.x };
    sws_scale(sws_ctx, src, src_stride, 0, wh.y, yuvpic->data, yuvpic->linesize);

    yuvpic->pts = outframe++;
    encode_and_write_frame(yuvpic);
}

CpuVideoWriter::~CpuVideoWriter() {
    cout << "Cleaning up CpuVideoWriter..." << endl;

    while(encode_and_write_frame(NULL));

    avcodec_free_context(&videoCodecContext);
    av_frame_free(&yuvpic);
    av_packet_free(&pkt);
    sws_freeContext(sws_ctx);

    av_write_trailer(fc);
    avio_closep(&fc->pb);
    avformat_free_context(fc);
}
