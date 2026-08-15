#include "VideoWriter.h"
#include "../Core/State/GlobalState.h"
#include "../Core/Smoketest.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <unistd.h>
#include <chrono>
#include "Writer.h"
#include <cstdlib>

#if defined(USE_NVIDIA)
    #define PIXEL_FORMAT AV_PIX_FMT_CUDA
    #define HWDEVICE_TYPE AV_HWDEVICE_TYPE_CUDA
    #define CODEC_NAME "hevc_nvenc"
#elif defined(USE_AMD)
    #define PIXEL_FORMAT AV_PIX_FMT_VAAPI
    #define HWDEVICE_TYPE AV_HWDEVICE_TYPE_VAAPI
    #define CODEC_NAME "av1_vaapi"
#else // Placeholders, shouldn't actually happen
    #define PIXEL_FORMAT AV_PIX_FMT_YUV420P
    #define HWDEVICE_TYPE AV_HWDEVICE_TYPE_NONE
    #define CODEC_NAME "libx265"
#endif

using namespace std;
extern "C" void preprocess_argb_to_p010(
    const uint32_t* d_argb,
    uint16_t* d_y_plane,
    uint16_t* d_uv_plane,
    int fd,
    size_t obj_size,
    int width,
    int height,
    int y_pitch_bytes,
    int uv_pitch_bytes,
    unsigned long long y_offset,
    unsigned long long uv_offset,
    uint32_t bg
);
extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

const bool USE_LIVE = false;
bool VideoWriter::encode_and_write_frame(AVFrame* frame){
    int ret = avcodec_send_frame(videoCodecContext, frame);
    //if (ret == AVERROR_EOF) return false;
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw runtime_error(string("avcodec_send_frame failed: ") + errbuf);
    }

    while (true) {
        int ret2 = avcodec_receive_packet(videoCodecContext, &pkt);
        if (ret2 == 0) {
            av_packet_rescale_ts(&pkt, videoCodecContext->time_base, videoStream->time_base);
            pkt.stream_index = videoStream->index;
            av_interleaved_write_frame(fc, &pkt);
            av_packet_unref(&pkt);
        } else if (ret2 == AVERROR(EAGAIN)) {
            return true;
        } else if (ret2 == AVERROR_EOF) {
            cout << "Encoder flushed, no more packets to receive." << endl;
            return false;
        } else
            throw runtime_error("Failed to receive video packet!");
    }

    return true;
}

VideoWriter::VideoWriter(AVFormatContext *fc_, const string& video_path, int video_width_pixels, int video_height_pixels, int video_framerate_fps) : fc(fc_) {
    if(USE_LIVE) {
        lp = new LivePlayer(ivec2(video_width_pixels, video_height_pixels));
        return;
    }

    #ifdef USE_AMD
    setenv("AMD_DEBUG", "notiling", 1);
    #endif
    av_log_set_level(AV_LOG_DEBUG);

    // Setting up the codec.
    const AVCodec* codec = avcodec_find_encoder_by_name(CODEC_NAME);
    if (!codec) {
        throw runtime_error("Failed to find video codec");
    }

    AVBufferRef* hw_device_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, HWDEVICE_TYPE, nullptr, nullptr, 0);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        cout << "Failed to create CUDA device context: " << errbuf << endl;
        throw runtime_error("Failed to create CUDA device context!");
    }

    videoStream = avformat_new_stream(fc, codec);
    if (!videoStream) {
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed to create new videostream!");
    }

    videoCodecContext = avcodec_alloc_context3(codec);
    if (!videoCodecContext) {
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed to allocate video codec context.");
    }
    videoCodecContext->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    videoCodecContext->width = video_width_pixels;
    videoCodecContext->height = video_height_pixels;
    videoCodecContext->pix_fmt = PIXEL_FORMAT;
    videoCodecContext->colorspace = AVCOL_SPC_BT709;
    videoCodecContext->color_primaries = AVCOL_PRI_BT709;
    videoCodecContext->color_trc = AVCOL_TRC_BT709;
    videoCodecContext->time_base = { 1, video_framerate_fps };
    videoCodecContext->framerate = { video_framerate_fps, 1 };
    videoCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    videoCodecContext->hw_frames_ctx = av_hwframe_ctx_alloc(videoCodecContext->hw_device_ctx);
    if (!videoCodecContext->hw_frames_ctx) {
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed to allocate hardware frame context!");
    }

    AVHWFramesContext* frames_ctx = reinterpret_cast<AVHWFramesContext*>(videoCodecContext->hw_frames_ctx->data);
    frames_ctx->format = videoCodecContext->pix_fmt;
    frames_ctx->sw_format = AV_PIX_FMT_P010LE;
    frames_ctx->width = video_width_pixels;
    frames_ctx->height = video_height_pixels;
    frames_ctx->initial_pool_size = 4;

    ret = av_hwframe_ctx_init(videoCodecContext->hw_frames_ctx);
    if (ret < 0) {
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed to initialize hardware frame context!");
    }
    
    // Sets quality compatible with both hevc and av1, extra options are ignored
    AVDictionary* opt = NULL;
    av_dict_set(&opt, "qp", "20", 0);
    av_dict_set(&opt, "global_quality", "20", 0);

    int ret2 = avcodec_open2(videoCodecContext, codec, &opt);
    if (ret2 < 0) {
        char errbuf[256];
        av_strerror(ret2, errbuf, sizeof(errbuf));
        cout << "Failed to open video codec " << CODEC_NAME << ": " << errbuf << endl;
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed avcodec_open2!");
    }
    av_dict_free(&opt);

    ret = avcodec_parameters_from_context(videoStream->codecpar, videoCodecContext);
    if (ret < 0) {
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed avcodec_parameters_from_context!");
    }

    ret = avio_open(&fc->pb, video_path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed avio_open!");
    }

    ret = avformat_write_header(fc, &opt);
    if (ret < 0) {
        cout << "Failed to write header: " << ff_errstr(ret) << endl;
        av_buffer_unref(&hw_device_ctx);
        throw runtime_error("Failed to write header!");
    }

    av_buffer_unref(&hw_device_ctx);
}

void VideoWriter::add_frame(uint32_t* device_pixels) {
    bool live = rendering_on();

    static auto last_print_time = chrono::steady_clock::time_point::min();
    auto now = chrono::steady_clock::now();
    if(!live || last_print_time == chrono::steady_clock::time_point::min() || chrono::duration_cast<chrono::seconds>(now - last_print_time).count() >= 1) {
        Pixels p(get_video_dimensions_pixels());
        cuda_copy_pixels_to_host(p.pixels.data(), get_video_width_pixels() * get_video_height_pixels(), device_pixels);
        p.print_to_terminal();
        last_print_time = now;
    }

    if (!live) return; // Don't encode video in smoketest

    if(USE_LIVE) {
        lp->accept_frame(device_pixels, false);
        return;
    }

    AVFrame* gpu_frame = av_frame_alloc();
    if (!gpu_frame) {
        throw runtime_error("Failed to allocate frame!");
    }

    gpu_frame->format = PIXEL_FORMAT;
    gpu_frame->width  = get_video_width_pixels();
    gpu_frame->height = get_video_height_pixels();
    gpu_frame->hw_frames_ctx = av_buffer_ref(videoCodecContext->hw_frames_ctx);

    int ret = av_hwframe_get_buffer(videoCodecContext->hw_frames_ctx, gpu_frame, 0);
    if (ret < 0) {
        av_frame_free(&gpu_frame);
        throw runtime_error("Failed to allocate hardware frame buffer!");
    }
    
    // Initialize values only used on AMD
    int fd = 0;
    size_t obj_size = 0;
    unsigned long long y_offset = 0;
    unsigned long long uv_offset = 0;
    
    int y_pitch = gpu_frame->linesize[0];
    int uv_pitch = gpu_frame->linesize[1];

    #ifdef USE_AMD
    AVFrame* drm_frame = av_frame_alloc();
    if (!drm_frame) {
        av_frame_free(&gpu_frame);
        av_frame_free(&drm_frame);
        throw runtime_error("Failed to allocate DRM frame!");
    }
    drm_frame->format = AV_PIX_FMT_DRM_PRIME;

    int ret2 = av_hwframe_map(drm_frame, gpu_frame, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_WRITE);
    if (ret2 < 0) {
        av_frame_free(&gpu_frame);
        av_frame_free(&drm_frame);
        throw runtime_error("Failed to map VAAPI surface to DRM PRIME!");
    };

    AVDRMFrameDescriptor* desc = (AVDRMFrameDescriptor*) drm_frame->data[0];
    
    int obj_idx = desc->layers[0].planes[0].object_index;

    fd      = desc->objects[obj_idx].fd;
    obj_size = desc->objects[obj_idx].size;
    y_pitch = desc->layers[0].planes[0].pitch;
    uv_pitch = desc->layers[1].planes[0].pitch;
    y_offset = desc->layers[0].planes[0].offset;
    uv_offset = desc->layers[1].planes[0].offset;
    #endif

    preprocess_argb_to_p010(
            device_pixels,
            reinterpret_cast<uint16_t*>(gpu_frame->data[0]), // NULL on AMD, properly filled inside CUDA call
            reinterpret_cast<uint16_t*>(gpu_frame->data[1]), // NULL on AMD, properly filled inside CUDA call
            fd, // used by AMD only
            obj_size, // used by AMD only
            gpu_frame->width,
            gpu_frame->height,
            y_pitch,
            uv_pitch,
            y_offset, // used by AMD only
            uv_offset, // used by AMD only
            get_video_background_color()
    );

    gpu_frame->pts = outframe++;

    encode_and_write_frame(gpu_frame);

    #ifdef USE_AMD
    av_frame_free(&drm_frame);
    #endif
    av_frame_free(&gpu_frame);
}

VideoWriter::~VideoWriter() {
    if(USE_LIVE) {
        delete lp;
        return;
    }

    cout << "Cleaning up VideoWriter..." << endl;

    while(encode_and_write_frame(NULL));

    avcodec_free_context(&videoCodecContext);

    av_packet_unref(&pkt);

    av_write_trailer(fc);
    avio_closep(&fc->pb);
    avformat_free_context(fc);
}
