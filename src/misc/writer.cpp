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

        // Preparing the containers of the frame data:
        // Allocating memory for each RGB frame, which will be converted to YUV.
        rgbpic = av_frame_alloc();
        rgbpic->format = AV_PIX_FMT_RGB24;
        rgbpic->width = width;
        rgbpic->height = height;
        av_frame_get_buffer(rgbpic, 1);

        // Allocating memory for each conversion output YUV frame.
        yuvpic = av_frame_alloc();
        yuvpic->format = AV_PIX_FMT_YUV420P;
        yuvpic->width = width;
        yuvpic->height = height;
        av_frame_get_buffer(yuvpic, 1);
    }

    void addAudioFrame(AVFrame* frame) {
        cout << "aaa" << endl;
        int ret = avcodec_send_frame(c, frame);
        if (ret < 0) {
            // Error handling
            cout << "Error sending audio frame to encoder: " << ret << endl;
            exit(1);
        }cout << "bbb" << endl;

        while (ret >= 0) {
            AVPacket pkt;
            av_init_packet(&pkt);

            ret = avcodec_receive_packet(c, &pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                // Error handling
                cout << "Error receiving audio packet from encoder: " << ret << endl;
                exit(1);
            }

            av_packet_rescale_ts(&pkt, c->time_base, stream->time_base);
            pkt.stream_index = stream->index;

            ret = av_interleaved_write_frame(fc, &pkt);
            if (ret < 0) {
                // Error handling
                cout << "Error writing audio packet to output file: " << ret << endl;
                exit(1);
            }

            av_packet_unref(&pkt);
        }
    }

    void add_audio_from_file(const std::string& filename) {
        AVFormatContext* inputFormatContext = nullptr;
        AVCodecContext* audioDecoderContext = nullptr;
        AVPacket packet;

        int ret = avformat_open_input(&inputFormatContext, filename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            // Error handling
            cout << "Could not open input file: " << ret << endl;
            exit(1);
        }

        ret = avformat_find_stream_info(inputFormatContext, nullptr);
        if (ret < 0) {
            // Error handling
            cout << "Could not find stream information: " << ret << endl;
            exit(1);
        }

        int audioStreamIndex = av_find_best_stream(inputFormatContext, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (audioStreamIndex < 0) {
            // Error handling
            cout << "Could not find audio stream in input file" << endl;
            exit(1);
        }

        AVCodec* audioDecoder = avcodec_find_decoder(inputFormatContext->streams[audioStreamIndex]->codec->codec_id);
        if (!audioDecoder) {
            // Error handling
            cout << "Could not find audio decoder" << endl;
            exit(1);
        }

        audioDecoderContext = avcodec_alloc_context3(audioDecoder);
        if (!audioDecoderContext) {
            // Error handling
            cout << "Could not allocate audio decoder context" << endl;
            exit(1);
        }

        ret = avcodec_parameters_to_context(audioDecoderContext, inputFormatContext->streams[audioStreamIndex]->codec);
        if (ret < 0) {
            // Error handling
            cout << "Could not initialize audio decoder context: " << ret << endl;
            exit(1);
        }

        ret = avcodec_open2(audioDecoderContext, audioDecoder, nullptr);
        if (ret < 0) {
            // Error handling
            cout << "Could not open audio decoder: " << ret << endl;
            exit(1);
        }

        AVFrame* audioFrame = av_frame_alloc();
        if (!audioFrame) {
            // Error handling
            cout << "Could not allocate audio frame" << endl;
            exit(1);
        }

        while (av_read_frame(inputFormatContext, &packet) >= 0) {
            if (packet.stream_index == audioStreamIndex) {
                ret = avcodec_send_packet(audioDecoderContext, &packet);
                if (ret < 0) {
                    // Error handling
                    cout << "Error sending audio packet to decoder: " << ret << endl;
                    exit(1);
                }

                while (ret >= 0) {
                    ret = avcodec_receive_frame(audioDecoderContext, audioFrame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        // Error handling
                        cout << "Error receiving audio frame from decoder: " << ret << endl;
                        exit(1);
                    }

                    addAudioFrame(audioFrame);
                }
            }

            av_packet_unref(&packet);
        }

        ret = avcodec_send_packet(audioDecoderContext, nullptr);
        while (ret >= 0) {
            ret = avcodec_receive_frame(audioDecoderContext, audioFrame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                // Error handling
                cout << "Error receiving audio frame from decoder: " << ret << endl;
                exit(1);
            }

            addAudioFrame(audioFrame);
        }

        av_frame_free(&audioFrame);
        avcodec_free_context(&audioDecoderContext);
        avformat_close_input(&inputFormatContext);
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
        av_write_trailer(fc);

        // Closing the file.
        avio_closep(&fc->pb);
        avcodec_close(stream->codecpar);

        // Freeing all the allocated memory:
        sws_freeContext(sws_ctx);
        av_frame_free(&rgbpic);
        av_frame_free(&yuvpic);

        avformat_free_context(fc);
    }
};
