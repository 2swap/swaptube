// Adapted from https://stackoverflow.com/questions/34511312

#pragma once

#include "pixels.h"

extern "C"
{
    #include <libswscale/swscale.h>
    #include <libavformat/avformat.h>
}

using namespace std;

class MovieWriter
{
    const unsigned int width, height;
    unsigned int inframe, outframe, audframe;
    int framerate;

    SwsContext* sws_ctx;
    AVStream* stream;
    AVFormatContext* fc;
    AVCodecContext* c;
    AVPacket pkt;

    AVFrame *rgbpic;
    AVFrame *yuvpic;

public:

    MovieWriter(const std::string& filename_, const unsigned int width_, const unsigned int height_, const int framerate_) :
        
    width(width_), height(height_), inframe(0), outframe(0), audframe(0), framerate(framerate_)

    {
        // Preparing to convert my generated RGB images to YUV frames.
        sws_ctx = sws_getContext(width, height,
            AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);

        // Preparing the data concerning the format and codec,
        // in order to write properly the header, frame data and end of file.
        const string filename = filename_ + ".mp4";
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
        c->time_base = { 1, framerate };

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
        stream->time_base = { 1, framerate };
        av_dump_format(fc, 0, filename.c_str(), 1);
        avio_open(&fc->pb, filename.c_str(), AVIO_FLAG_WRITE);
        avformat_write_header(fc, &opt);
        av_dict_free(&opt);

        // Allocating memory for each RGB and YUV frame.
        rgbpic = av_frame_alloc();
        yuvpic = av_frame_alloc();
        rgbpic->format = AV_PIX_FMT_RGB24;
        yuvpic->format = AV_PIX_FMT_YUV420P;
        rgbpic->width = yuvpic->width = width;
        rgbpic->height = yuvpic->height = height;
        av_frame_get_buffer(rgbpic, 1);
        av_frame_get_buffer(yuvpic, 1);

        avcodec_register_all();
    }




    bool encode_and_write_frame(AVFrame* frame){
        int got_output = 0;
        avcodec_encode_video2(c, &pkt, frame, &got_output);
        if (!got_output) return false;

        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, { 1, framerate }, stream->time_base);

        pkt.stream_index = stream->index;
        cout << "Writing frame " << outframe++ << " (size = " << pkt.size << ")" << endl;

        // Write the encoded frame to the mp4 file.
        av_interleaved_write_frame(fc, &pkt);
        av_packet_unref(&pkt);

        return true;
    }

    void addFrame(const Pixels& p)
    {
        cout << "Encoding frame " << inframe++ << ". ";
        const uint8_t* pixels = &p.pixels[0];

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
        sws_scale(sws_ctx, rgbpic->data, rgbpic->linesize, 0, height, yuvpic->data, yuvpic->linesize);

        av_init_packet(&pkt);
        pkt.data = NULL;
        pkt.size = 0;

        // The PTS of the frame are just in a reference unit,
        // unrelated to the format we are using. We set them,
        // for instance, as the corresponding frame number.
        yuvpic->pts = outframe;

        if(!encode_and_write_frame(yuvpic)) cout << endl;
    }

    ~MovieWriter()
    {
        // Writing the delayed frames
        while(encode_and_write_frame(NULL));
        
        // Writing the end of the file.
        cout << "writing file ending" << endl;
        av_write_trailer(fc);

        // Closing the file.
        cout << "closing the file" << endl;
        avio_closep(&fc->pb);
        avcodec_close(stream->codec);

        // Freeing all the allocated memory:
        cout << "freeing some stuff" << endl;
        sws_freeContext(sws_ctx);
        av_frame_free(&rgbpic);
        av_frame_free(&yuvpic);

        cout << "freeing context" << endl;
        avformat_free_context(fc);
    }
};
