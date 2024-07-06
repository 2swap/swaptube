#pragma once
#include <iostream>
#include <string>
extern "C"
{
    #include <libswscale/swscale.h>
    #include <libavformat/avformat.h>
}

using namespace std;

class VideoWriter {
private:
    string output_filename;

    //note that the FormatContext has shared ownership with audioWriter and MovieWriter
    AVFormatContext *fc = nullptr;

    SwsContext *sws_ctx = nullptr;
    AVStream *videoStream = nullptr;
    AVCodecContext *videoCodecContext = nullptr;
    AVFrame *rgbpic = nullptr;
    AVFrame *yuvpic = nullptr;
    AVPacket pkt = {0};
    unsigned outframe = 0;

    bool encode_and_write_frame(AVFrame* frame){
        if(frame != NULL){
            int ret = avcodec_send_frame(videoCodecContext, frame);
            if (ret<0) {
                cout << "Failed encoding video!" << endl;
                exit(1);
            }
        }

        int ret2 = avcodec_receive_packet(videoCodecContext, &pkt);
        if (ret2 == AVERROR(EAGAIN) || ret2 == AVERROR_EOF) {
            return false;
        } else if (ret2!=0) {
            cout << "Failed to receive video packet!" << endl;
            exit(1);
        }

        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, { 1, VIDEO_FRAMERATE }, videoStream->time_base);

        pkt.stream_index = videoStream->index;

        // Write the encoded frame to the mp4 file.
        av_interleaved_write_frame(fc, &pkt);

        return true;
    }

public:
    VideoWriter(const string& _output_filename, AVFormatContext *fc_)
        : output_filename(_output_filename), fc(fc_) {}

    void init_video() {
        av_log_set_level(AV_LOG_DEBUG);

        // Preparing to convert my generated RGB images to YUV frames.
        sws_ctx = sws_getContext(VIDEO_WIDTH, VIDEO_HEIGHT, AV_PIX_FMT_RGB24,
                                 VIDEO_WIDTH, VIDEO_HEIGHT, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);

        // Setting up the codec.
        AVCodec* codec = avcodec_find_encoder_by_name("libx264");
        AVDictionary* opt = NULL;
        av_dict_set(&opt, "preset", "ultrafast", 0);
        av_dict_set(&opt, "crf", "18", 0);

        videoStream = avformat_new_stream(fc, codec);
        if (!videoStream) {
            cout << "Failed create new videostream!" << endl;
            exit(1);
        }

        videoCodecContext = avcodec_alloc_context3(codec);
        videoCodecContext->width = VIDEO_WIDTH;
        videoCodecContext->height = VIDEO_HEIGHT;
        videoCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
        videoCodecContext->time_base = { 1, VIDEO_FRAMERATE };

        avcodec_parameters_from_context(videoStream->codecpar, videoCodecContext);

        avcodec_open2(videoCodecContext, codec, &opt);
        av_dict_free(&opt);

        videoStream->time_base = { 1, VIDEO_FRAMERATE };
        av_dump_format(fc, 0, output_filename.c_str(), 1);
        avio_open(&fc->pb, output_filename.c_str(), AVIO_FLAG_WRITE);
        int ret = avformat_write_header(fc, &opt);
        if (ret < 0) {
            cout << "Failed to write header!" << endl;
            exit(1);
        }
        av_dict_free(&opt);

        // Allocating memory for each RGB and YUV frame.
        rgbpic = av_frame_alloc();
        yuvpic = av_frame_alloc();
        rgbpic->format = AV_PIX_FMT_RGB24;
        yuvpic->format = AV_PIX_FMT_YUV420P;
        rgbpic->width = yuvpic->width = VIDEO_WIDTH;
        rgbpic->height = yuvpic->height = VIDEO_HEIGHT;
        av_frame_get_buffer(rgbpic, 1);
        av_frame_get_buffer(yuvpic, 1);
    }
    void add_frame(const Pixels& p) {
        // The AVFrame data will be stored as RGBRGBRGB... row-wise,
        // from left to right and from top to bottom.
        for (unsigned int y = 0; y < VIDEO_HEIGHT; y++)
        {
            // rgbpic->linesize[0] is equal to width.
            int rowboi = y * rgbpic->linesize[0];
            for (unsigned int x = 0; x < VIDEO_WIDTH; x++)
            {
                int a,r,g,b;
                p.get_pixel_by_channels(x, y, a, r, g, b);
                double alpha = a/255.; // in the end, we pretend there is a black background
                rgbpic->data[0][rowboi + 3 * x + 0] = r*alpha;
                rgbpic->data[0][rowboi + 3 * x + 1] = g*alpha;
                rgbpic->data[0][rowboi + 3 * x + 2] = b*alpha;
            }
        }

        // Not actually scaling anything, but just converting
        // the RGB data to YUV and store it in yuvpic.
        sws_scale(sws_ctx, rgbpic->data, rgbpic->linesize, 0, VIDEO_HEIGHT, yuvpic->data, yuvpic->linesize);

        // The PTS of the frame are just in a reference unit,
        // unrelated to the format we are using. We set them,
        // for instance, as the corresponding frame number.
        yuvpic->pts = outframe;
        outframe++;

        if(!encode_and_write_frame(yuvpic)) cout << endl;
    }
    void cleanup() {
        // Writing the delayed video frames
        while(encode_and_write_frame(NULL));

        // Closing the video codec.
        avcodec_close(videoCodecContext);

        // Freeing video specific resources
        cout << "freeing some stuff" << endl;
        sws_freeContext(sws_ctx);
        av_frame_free(&rgbpic);
        av_frame_free(&yuvpic);

        av_packet_unref(&pkt);

        cout << "Done video cleanup" << endl;
    }
};
