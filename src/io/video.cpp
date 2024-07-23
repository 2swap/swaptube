#pragma once
#include <iostream>
#include <string>
#include <regex>
#include <cassert>
extern "C"
{
    #include <libswscale/swscale.h>
    #include <libavformat/avformat.h>
}

using namespace std;

// Function to redirect stderr to a pipe
int redirect_stderr(int pipefd[2]) {
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return -1;
    }

    // Save the original stderr
    int original_stderr = dup(STDERR_FILENO);

    // Redirect stderr to the write end of the pipe
    if (dup2(pipefd[1], STDERR_FILENO) == -1) {
        perror("dup2");
        return -1;
    }

    // Close the write end of the pipe in the parent process
    close(pipefd[1]);

    return original_stderr;
}

// Function to restore the original stderr
void restore_stderr(int original_stderr) {
    dup2(original_stderr, STDERR_FILENO);
    close(original_stderr);
}

// Function to read from a file descriptor into a string
string read_from_fd(int fd) {
    string output;
    char buffer[1024];
    ssize_t bytes_read;

    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        output.append(buffer, bytes_read);
    }

    return output;
}

// Function to parse the debug output
void parse_debug_output(const string& output) {
    istringstream iss(output);
    string line;
    regex regex_pattern(R"(frame=\s*(\d+)\s*QP=(\d+\.\d+)\s+NAL=(\d+)\s+Slice:([IPBS])\s+Poc:(\d+)\s+I:(\d+)\s+P:(\d+)\s+SKIP:(\d+)\s+size=(\d+)\s+bytes)");
    regex regex_pattern_empty_line(R"(\s*)");
    while (getline(iss, line)) {
        smatch match;
        if (regex_search(line, match, regex_pattern) && match.size() == 10) {
            int frame = stoi(match[1].str());
            float qp = stof(match[2].str());
            int nal = stoi(match[3].str());
            string slice = match[4].str();
            int poc = stoi(match[5].str());
            int i = stoi(match[6].str());
            int p = stoi(match[7].str());
            int skip = stoi(match[8].str());
            int size = stoi(match[9].str());
        } else if(regex_search(line, match, regex_pattern_empty_line)) {
            // do nothing
        } else {
            // If the string did not match the expected format, dump it to stderr
            cerr << "Failed to parse cerr output from encoder: " << line << endl;
        }
    }
}

/*
 * Wrapper around avcodec_send_frame which captures its output to stderr.
 */
int send_frame(AVCodecContext* vcc, AVFrame* frame){
    int pipefd[2];
    int original_stderr = redirect_stderr(pipefd);
    int ret = avcodec_send_frame(vcc, frame);
    restore_stderr(original_stderr);
    string debug_output = read_from_fd(pipefd[0]);
    close(pipefd[0]);
    parse_debug_output(debug_output);
    return ret;
}

class VideoWriter {
private:
    const int VIDEO_WIDTH;
    const int VIDEO_HEIGHT;
    const int VIDEO_FRAMERATE;
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
            int ret = send_frame(videoCodecContext, frame);
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
    VideoWriter(int w, int h, int fr, const string& output_folder, const string& _output_filename, AVFormatContext *fc_)
        : VIDEO_WIDTH(w), VIDEO_HEIGHT(h), VIDEO_FRAMERATE(fr),
        output_filename(output_folder + _output_filename), fc(fc_) {
        ensure_directory_exists(output_folder);
    }

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
        assert(p.w == VIDEO_WIDTH && p.h == VIDEO_HEIGHT);
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

        encode_and_write_frame(yuvpic);
    }
    void cleanup() {
        // Writing the delayed video frames
        while(encode_and_write_frame(NULL));

        // Closing the video codec.
        avcodec_close(videoCodecContext);

        // Freeing video specific resources
        sws_freeContext(sws_ctx);
        av_frame_free(&rgbpic);
        av_frame_free(&yuvpic);

        av_packet_unref(&pkt);

        av_write_trailer(fc);
        avio_closep(&fc->pb);
        avformat_free_context(fc);
    }
};
