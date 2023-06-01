// Adapted from https://stackoverflow.com/questions/34511312/how-to-encode-a-video-from-several-images-generated-in-a-c-program-without-wri
// All credit to the authors of https://github.com/apc-llc/moviemaker-cpp. Slightly modified for context-specificity.

#pragma once

#include <librsvg-2.0/librsvg/rsvg.h>
#include <vector>
#include <iostream>
#include <memory>
#include "pixels.h"

extern "C"
{
    #include <x264.h>
    #include <libswscale/swscale.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/mathematics.h>
    #include <libavformat/avformat.h>
    #include <libavutil/opt.h>
}

using namespace std;

// One-time initialization.
class FFmpegInitialize
{
public :

    FFmpegInitialize()
    {
        // Loads the whole database of available codecs and formats.
        av_register_all();
    }
};

static FFmpegInitialize ffmpegInitialize;

class MovieWriter
{
    const unsigned int width, height;
    unsigned int iframe;
    int framerate;

    SwsContext* swsCtx;
    AVOutputFormat* fmt;
    AVStream* stream;
    AVFormatContext* fc;
    AVCodecContext* c;
    AVPacket pkt;

    AVFrame *rgbpic, *yuvpic;

    std::vector<uint8_t> pixels;

    cairo_surface_t* cairo_surface;

public :

    MovieWriter(const std::string& filename_, const unsigned int width_, const unsigned int height_, const int framerate_ = 25) :
        
    width(width_), height(height_), iframe(0), framerate(framerate_),
          pixels(4 * width * height)

    {
        cairo_surface = cairo_image_surface_create_for_data(
            (unsigned char*)&pixels[0], CAIRO_FORMAT_RGB24, width, height,
            cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, width));

        // Preparing to convert my generated RGB images to YUV frames.
        swsCtx = sws_getContext(width, height,
            AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);

        // Preparing the data concerning the format and codec,
        // in order to write properly the header, frame data and end of file.
        const char* fmtext = "mp4";
        const string filename = filename_ + "." + fmtext;
        fmt = av_guess_format(fmtext, NULL, NULL);
        avformat_alloc_output_context2(&fc, NULL, NULL, filename.c_str());

        // Setting up the codec.
        AVCodec* codec = avcodec_find_encoder_by_name("libx264");
        AVDictionary* opt = NULL;
        av_dict_set(&opt, "preset", "ultrafast", 0);
        av_dict_set(&opt, "crf", "23", 0);
        stream = avformat_new_stream(fc, codec);
        c = stream->codec;
        c->width = width;
        c->height = height;
        c->pix_fmt = AV_PIX_FMT_YUV420P;
        c->time_base = (AVRational){ 1, framerate };

        // Setting up the format, its stream(s),
        // linking with the codec(s) and write the header.
        if (fc->oformat->flags & AVFMT_GLOBALHEADER)
        {
            // Some formats require a global header.
            c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        avcodec_open2(c, codec, &opt);
        av_dict_free(&opt);

        // Once the codec is set up, we need to let the container know
        // which codec are the streams using, in this case the only (video) stream.
        stream->time_base = (AVRational){ 1, framerate };
        av_dump_format(fc, 0, filename.c_str(), 1);
        avio_open(&fc->pb, filename.c_str(), AVIO_FLAG_WRITE);
        int ret = avformat_write_header(fc, &opt);
        av_dict_free(&opt); 

        // Preparing the containers of the frame data:
        // Allocating memory for each RGB frame, which will be lately converted to YUV.
        rgbpic = av_frame_alloc();
        rgbpic->format = AV_PIX_FMT_RGB24;
        rgbpic->width = width;
        rgbpic->height = height;
        ret = av_frame_get_buffer(rgbpic, 1);

        // Allocating memory for each conversion output YUV frame.
        yuvpic = av_frame_alloc();
        yuvpic->format = AV_PIX_FMT_YUV420P;
        yuvpic->width = width;
        yuvpic->height = height;
        ret = av_frame_get_buffer(yuvpic, 1);
    }

    void addFrame(const uint8_t* pixels)
    {
        // The AVFrame data will be stored as RGBRGBRGB... row-wise,
        // from left to right and from top to bottom.
        for (unsigned int y = 0; y < height; y++)
        {
            for (unsigned int x = 0; x < width; x++)
            {
                // rgbpic->linesize[0] is equal to width.
                rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 0] = pixels[y * 4 * width + 4 * x + 2];
                rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 1] = pixels[y * 4 * width + 4 * x + 1];
                rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 2] = pixels[y * 4 * width + 4 * x + 0];
            }
        }

        // Not actually scaling anything, but just converting
        // the RGB data to YUV and store it in yuvpic.
        sws_scale(swsCtx, rgbpic->data, rgbpic->linesize, 0,
            height, yuvpic->data, yuvpic->linesize);

        av_init_packet(&pkt);
        pkt.data = NULL;
        pkt.size = 0;

        // The PTS of the frame are just in a reference unit,
        // unrelated to the format we are using. We set them,
        // for instance, as the corresponding frame number.
        yuvpic->pts = iframe;

        int got_output;
        int ret = avcodec_encode_video2(c, &pkt, yuvpic, &got_output);
        if (got_output)
        {
            fflush(stdout);

            // We set the packet PTS and DTS taking in the account our FPS (second argument),
            // and the time base that our selected format uses (third argument).
            av_packet_rescale_ts(&pkt, (AVRational){ 1, framerate }, stream->time_base);

            pkt.stream_index = stream->index;
            cout << "Writing frame " << iframe++ << " (size = " << pkt.size << ")" << endl;

            // Write the encoded frame to the mp4 file.
            av_interleaved_write_frame(fc, &pkt);
            av_packet_unref(&pkt);
        }
    }

    ~MovieWriter()
    {
        // Writing the delayed frames:
        for (int got_output = 1; got_output; )
        {
            int ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
            if (got_output)
            {
                fflush(stdout);
                av_packet_rescale_ts(&pkt, (AVRational){ 1, framerate }, stream->time_base);
                pkt.stream_index = stream->index;
                cout << "Writing frame " << iframe++ << " (size = " << pkt.size << ")" << endl;
                av_interleaved_write_frame(fc, &pkt);
                av_packet_unref(&pkt);
            }
        }
        
        // Writing the end of the file.
        av_write_trailer(fc);

        // Closing the file.
        if (!(fmt->flags & AVFMT_NOFILE))
            avio_closep(&fc->pb);
        avcodec_close(stream->codec);

        // Freeing all the allocated memory:
        sws_freeContext(swsCtx);
        av_frame_free(&rgbpic);
        av_frame_free(&yuvpic);
        avformat_free_context(fc);

        cairo_surface_destroy(cairo_surface);
    }
};
