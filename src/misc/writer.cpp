// Adapted from https://stackoverflow.com/questions/34511312

#pragma once
#include "pixels.h"

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
    unsigned int inframe, outframe, audframe, audiodts, audiopts;
    int framerate;
    int audioStreamIndex;

    string output_filename;

    SwsContext* sws_ctx;
    AVStream* videoStream;
    AVStream* audioStream;
    AVFormatContext* fc;
    AVCodecContext* videoCodecContext;
    AVCodecContext* audioInputCodecContext;
    AVCodecContext* audioOutputCodecContext;
    AVPacket pkt, inputPacket, outputPacket;

    AVFrame *rgbpic;
    AVFrame *yuvpic;

public:

    MovieWriter(const std::string& filename_, const unsigned int width_, const unsigned int height_, const int framerate_) :
        
    width(width_), height(height_), inframe(0), outframe(0), audframe(0), audiopts(0), audiodts(0), framerate(framerate_), output_filename(filename_),
    sws_ctx(nullptr), videoStream(nullptr), audioStream(nullptr), fc(nullptr), videoCodecContext(nullptr), rgbpic(nullptr), yuvpic(nullptr),
    pkt(), inputPacket(), outputPacket()

    {
    }

    void init(const string& inputAudioFilename){
        AVFormatContext* inputAudioFormatContext = nullptr;
        avformat_open_input(&inputAudioFormatContext, inputAudioFilename.c_str(), nullptr, nullptr);

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
        av_dict_set(&opt, "crf", "23", 0);
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

        AVPacket outputPacket;
        av_init_packet(&outputPacket);

        avcodec_register_all();
        
        avformat_close_input(&inputAudioFormatContext);
    }

    void add_audio(const string& inputAudioFilename) {
        cout << "Adding audio" << endl;

        AVFormatContext* inputAudioFormatContext = nullptr;
        avformat_open_input(&inputAudioFormatContext, inputAudioFilename.c_str(), nullptr, nullptr);

        // Read input audio frames and write to output format context
        while (av_read_frame(inputAudioFormatContext, &inputPacket) >= 0) {
            if (inputPacket.stream_index == audioStreamIndex) {
                AVFrame* frame = av_frame_alloc();
                int ret = avcodec_send_packet(audioInputCodecContext, &inputPacket);

                while (ret >= 0) {
                    ret = avcodec_receive_frame(audioInputCodecContext, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        cout << ret << " " << AVERROR(EAGAIN) << " " << AVERROR_EOF << endl;
                        break;
                    }

                    // Encode the audio frame
                    cout << "Frame " << audframe << endl;
                    frame->pts = audframe;
                    audframe++;
                    avcodec_send_frame(audioOutputCodecContext, frame);

                    while (ret >= 0) {
                        cout << err2str(ret) << endl;
                        ret = avcodec_receive_packet(audioOutputCodecContext, &outputPacket);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                            break;
                        }

                        // Set the stream index of the output packet to the audio stream index
                        outputPacket.stream_index = audioStream->index;

                        // Write the output packet to the output format context
                        outputPacket.dts = audiodts;
                        audiodts++;
                        outputPacket.pts = audiopts;
                        audiopts++;
                        ret = av_write_frame(fc, &outputPacket);

                        av_packet_unref(&outputPacket);
                    }
                }
                av_frame_unref(frame);
                av_frame_free(&frame);
            }
        }

        avformat_close_input(&inputAudioFormatContext);

        cout << "Audio added successfully" << endl;
    }

    void add_silence(double duration) {
        cout << "Adding silence: " << duration << " seconds" << endl;

        // Calculate the number of samples needed for the specified duration
        int numSamples = static_cast<int>(duration * audioOutputCodecContext->sample_rate);

        // Calculate the frame size based on the codec context's frame size
        int frameSize = audioOutputCodecContext->frame_size;

        // Calculate the number of frames needed to accommodate the specified duration
        int numFrames = (numSamples + frameSize - 1) / frameSize;

        // Allocate buffer for audio data
        int bufferSize = numSamples * audioOutputCodecContext->channels;
        vector<int16_t> audioBuffer(bufferSize, 0);

        // Split the audio data into multiple frames
        int samplesRemaining = numSamples;
        int samplesPerFrame = frameSize;

        for (int i = 0; i < numFrames; i++) {
            cout << "Frame " << audframe << endl;
            if (samplesRemaining < frameSize) {
                samplesPerFrame = samplesRemaining;
            }

            // Fill the audio buffer with silence
            for (int ch = 0; ch < audioOutputCodecContext->channels; ch++) {
                int offset = ch * samplesPerFrame;
                for (int s = 0; s < samplesPerFrame; s++) {
                    audioBuffer[offset + s] = s%5;
                }
            }

            // Create a frame and set its properties
            AVFrame* frame = av_frame_alloc();
            frame->nb_samples = samplesPerFrame;
            frame->channel_layout = audioOutputCodecContext->channel_layout;
            frame->sample_rate = audioOutputCodecContext->sample_rate;
            frame->format = audioOutputCodecContext->sample_fmt;

            // Fill the frame with the audio data
            int ret = av_frame_get_buffer(frame, 0);

            for (int ch = 0; ch < audioOutputCodecContext->channels; ch++) {
                frame->linesize[ch] = samplesPerFrame * av_get_bytes_per_sample(audioOutputCodecContext->sample_fmt);
            }

            ret = avcodec_fill_audio_frame(frame, audioOutputCodecContext->channels, audioOutputCodecContext->sample_fmt,
                                           reinterpret_cast<const uint8_t*>(audioBuffer.data()), bufferSize, 0);

            frame->pts = audframe;
            audframe++;
            ret = avcodec_send_frame(audioOutputCodecContext, frame);

            while (ret >= 0) {
                ret = avcodec_receive_packet(audioOutputCodecContext, &outputPacket);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    cout << err2str(ret) << endl;
                    break;
                }

                // Set the stream index of the output packet to the audio stream index
                outputPacket.stream_index = audioStream->index;

                outputPacket.dts = audiodts;
                audiodts++;
                outputPacket.pts = audiopts;
                audiopts++;
                // Write the output packet to the output format context
                ret = av_write_frame(fc, &outputPacket);

                av_packet_unref(&outputPacket);
            }

            av_frame_unref(frame);
            av_frame_free(&frame);

            samplesRemaining -= samplesPerFrame;
        }
        cout << "Added silence: " << duration << " seconds" << endl;
    }

    bool encode_and_write_frame(AVFrame* frame){
        int got_output = 0;
        avcodec_encode_video2(videoCodecContext, &pkt, frame, &got_output);
        if (!got_output) return false;

        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, { 1, framerate }, videoStream->time_base);

        pkt.stream_index = videoStream->index;
        cout << "Writing frame " << outframe++ << " (size = " << pkt.size << ")" << endl;

        // Write the encoded frame to the mp4 file.
        av_interleaved_write_frame(fc, &pkt);

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

        // The PTS of the frame are just in a reference unit,
        // unrelated to the format we are using. We set them,
        // for instance, as the corresponding frame number.
        yuvpic->pts = outframe;

        if(!encode_and_write_frame(yuvpic)) cout << endl;
    }

    ~MovieWriter()
    {


        // write delayed audio frames
        avcodec_send_frame(audioOutputCodecContext, NULL);
        int ret = 0;
        while (ret >= 0) {
            ret = avcodec_receive_packet(audioOutputCodecContext, &outputPacket);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }

            // Set the stream index of the output packet to the audio stream index
            outputPacket.stream_index = audioStream->index;

            // Write the output packet to the output format context
            ret = av_write_frame(fc, &outputPacket);

            av_packet_unref(&outputPacket);
        }


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
