// Adapted from https://stackoverflow.com/questions/34511312

#pragma once

#include <filesystem>
extern "C"
{
    #include <libswscale/swscale.h>
    #include <libavformat/avformat.h>
}

char* err2str(int errnum)
{
    // static char str[AV_ERROR_MAX_STRING_SIZE];
    // thread_local may be better than static in multi-thread circumstance
    thread_local char str[AV_ERROR_MAX_STRING_SIZE]; 
    memset(str, 0, sizeof(str));
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}


using namespace std;

class MovieWriter
{
    const unsigned int width, height;
    unsigned int inframe, outframe, audframe;
    int audiotime;
    int framerate;
    int audioStreamIndex;

    string output_filename;
    string media_folder;

    SwsContext* sws_ctx;
    AVStream* videoStream;
    AVStream* audioStream;
    AVFormatContext* fc;
    AVCodecContext* videoCodecContext;
    AVCodecContext* audioInputCodecContext;
    AVCodecContext* audioOutputCodecContext;
    AVPacket pkt, inputPacket;

    AVFrame *rgbpic;
    AVFrame *yuvpic;

public:

    void make_media_folder() {
        // Check if the folder already exists
        if (filesystem::exists(media_folder)) {
            cout << "Media folder already exists, not creating." << endl;
        } else {
            // Create the folder
            if (filesystem::create_directory(media_folder)) {
                cout << "Media folder created successfully." << endl;
            } else {
                cout << "Failed to create media folder." << endl;
            }
        }
    }

    MovieWriter(const string& filename_, const unsigned int width_, const unsigned int height_, const int framerate_, const string& media_) :
        
    width(width_), height(height_), inframe(0), outframe(0), audframe(0), framerate(framerate_), output_filename(filename_),
    sws_ctx(nullptr), videoStream(nullptr), audioStream(nullptr), fc(nullptr), videoCodecContext(nullptr), rgbpic(nullptr), yuvpic(nullptr),
    pkt(), inputPacket(), media_folder(media_), audiotime(-1)

    {
        make_media_folder();
    }

    double add_audio_get_length(const string& inputAudioFilename);
    void add_silence(double duration);
    double encode_and_write_audio();
    bool encode_and_write_frame(AVFrame* frame);
    void addFrame(const Pixels& p);
    void set_audiotime(double t_seconds);

    void init(const string& inputAudioFilename){
        AVFormatContext* inputAudioFormatContext = nullptr;
        avformat_open_input(&inputAudioFormatContext, inputAudioFilename.c_str(), nullptr, nullptr);
        cout << "Initializing writer with codec from " << inputAudioFilename << endl;

        // Find input audio stream information
        avformat_find_stream_info(inputAudioFormatContext, nullptr);

        // Find input audio stream
        audioStreamIndex = -1;
        AVCodecParameters* codecParams = nullptr;
        for (unsigned int i = 0; i < inputAudioFormatContext->nb_streams; ++i) {
            if (inputAudioFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndex = i;
                codecParams = inputAudioFormatContext->streams[i]->codecpar;
                break;
            }
        }

        // INPUT
        AVCodec* audioInputCodec = avcodec_find_decoder(codecParams->codec_id);
        audioInputCodecContext = avcodec_alloc_context3(audioInputCodec);
        avcodec_parameters_to_context(audioInputCodecContext, codecParams);
        avcodec_open2(audioInputCodecContext, audioInputCodec, nullptr);

        // Preparing to convert my generated RGB images to YUV frames.
        sws_ctx = sws_getContext(width, height,
            AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);

        // Preparing the data concerning the format and codec,
        // in order to write properly the header, frame data and end of file.
        avformat_alloc_output_context2(&fc, NULL, NULL, output_filename.c_str());

        // Setting up the codec.
        AVCodec* codec = avcodec_find_encoder_by_name("libx264");
        AVDictionary* opt = NULL;
        av_dict_set(&opt, "preset", "ultrafast", 0);
        av_dict_set(&opt, "crf", "18", 0);
        videoStream = avformat_new_stream(fc, codec);
        audioStream = avformat_new_stream(fc, audioInputCodec);
        videoCodecContext = videoStream->codec;
        videoCodecContext->width = width;
        videoCodecContext->height = height;
        videoCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
        videoCodecContext->time_base = { 1, framerate };


        // Create a new audio stream in the output format context
        // Copy codec parameters to the new audio stream
        avcodec_parameters_copy(audioStream->codecpar, codecParams);

        // OUTPUT
        AVCodec* audioOutputCodec = avcodec_find_encoder(codecParams->codec_id);
        audioOutputCodecContext = avcodec_alloc_context3(audioOutputCodec);
        avcodec_parameters_to_context(audioOutputCodecContext, codecParams);
        avcodec_open2(audioOutputCodecContext, audioOutputCodec, nullptr);

        // Setting up the format, its stream(s),
        // linking with the codec(s) and write the header.
        if (fc->oformat->flags & AVFMT_GLOBALHEADER)
        {
            // Some formats require a global header.
            videoCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        avcodec_open2(videoCodecContext, codec, &opt);
        av_dict_free(&opt);

        // Once the codec is set up, we need to let the container know
        // which codec are the streams using, in this case the only (video) stream.
        videoStream->time_base = { 1, framerate };
        av_dump_format(fc, 0, output_filename.c_str(), 1);
        avio_open(&fc->pb, output_filename.c_str(), AVIO_FLAG_WRITE);
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

        av_init_packet(&pkt);

        // Initialize input and output packets
        AVPacket inputPacket;
        av_init_packet(&inputPacket);

        avcodec_register_all();
        
        avformat_close_input(&inputAudioFormatContext);
    }

    ~MovieWriter()
    {
        // write delayed audio frames
        avcodec_send_frame(audioOutputCodecContext, NULL);
        encode_and_write_audio();

        // Writing the delayed video frames
        while(encode_and_write_frame(NULL));
        
        // Writing the end of the file.
        cout << "writing file ending" << endl;
        av_write_trailer(fc);

        // Closing the file.
        cout << "closing the file" << endl;
        avio_closep(&fc->pb);
        avcodec_close(videoStream->codec);

        // Freeing all the allocated memory:
        cout << "freeing some stuff" << endl;
        sws_freeContext(sws_ctx);
        av_frame_free(&rgbpic);
        av_frame_free(&yuvpic);

        cout << "freeing context" << endl;
        avformat_free_context(fc);

        avcodec_free_context(&audioOutputCodecContext);
        avcodec_free_context(&audioInputCodecContext);

        av_packet_unref(&inputPacket);
        av_packet_unref(&pkt);
    }
};

#include "audio.cpp"
#include "video.cpp"
