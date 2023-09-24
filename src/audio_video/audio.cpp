#pragma once
#include <sys/stat.h>
#include <vector>

void MovieWriter::set_audiotime(double t_seconds){
    double t_samples = audioOutputCodecContext->sample_rate * t_seconds;
    //if(t_samples < audiotime){
    //    cerr << "Audio PTS latchup!" << endl << "Was: " << audiotime << " and is being set to " << t_samples << "!" << endl << "Aborting!" << endl;
    //    exit(1);
    //}
    substime = t_seconds;
    audiotime = t_samples;
}

double MovieWriter::encode_and_write_audio(){
    AVPacket outputPacket;
    av_init_packet(&outputPacket);

    double length_in_seconds = 0;

    int ret = 1;
    while (ret >= 0) {
        ret = avcodec_receive_packet(audioOutputCodecContext, &outputPacket);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }

        // Set the stream index of the output packet to the audio stream index
        outputPacket.stream_index = audioStream->index;

        // Set the correct PTS and DTS values for the output packet
        outputPacket.dts = av_rescale_q(audiotime, audioStream->time_base, audioOutputCodecContext->time_base);
        outputPacket.pts = outputPacket.dts;
        audio_pts_file << outputPacket.pts << endl;
        audiotime += outputPacket.duration;

        length_in_seconds += static_cast<double>(outputPacket.duration) / audioOutputCodecContext->sample_rate;

        // Rescale PTS and DTS values before writing the packet
        av_packet_rescale_ts(&outputPacket, audioOutputCodecContext->time_base, audioStream->time_base);

        ret = av_write_frame(fc, &outputPacket);

        av_packet_unref(&outputPacket);
    }
    return length_in_seconds;
}

bool MovieWriter::file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

double MovieWriter::add_audio_get_length(const string& inputAudioFilename) {
    cout << "Adding audio" << endl;
    double length_in_seconds = 0;

    // Check if the input audio file exists
    std::string fullInputAudioFilename = media_folder + inputAudioFilename;
    if (!file_exists(fullInputAudioFilename)) {
        std::cerr << "Input audio file does not exist: " << fullInputAudioFilename << std::endl;
        length_in_seconds = 3.0;
        add_silence(length_in_seconds);
        return length_in_seconds;
    }

    AVFormatContext* inputAudioFormatContext = nullptr;
    int ret = avformat_open_input(&inputAudioFormatContext, fullInputAudioFilename.c_str(), nullptr, nullptr);
    if (ret < 0) {
        std::cerr << "Error opening input audio file." << std::endl;
        exit(1);
    }

    // Read input audio frames and write to output format context
    while (av_read_frame(inputAudioFormatContext, &inputPacket) >= 0) {
        if (inputPacket.stream_index == audioStreamIndex) {
            AVFrame* frame = av_frame_alloc();
            int ret = avcodec_send_packet(audioInputCodecContext, &inputPacket);

            while (ret >= 0) {
                ret = avcodec_receive_frame(audioInputCodecContext, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }

                frame->pts = audframe;
                audframe++;
                avcodec_send_frame(audioOutputCodecContext, frame);

                length_in_seconds += encode_and_write_audio();
            }
            av_frame_unref(frame);
            av_frame_free(&frame);
        }
    }

    avformat_close_input(&inputAudioFormatContext);

    cout << "Audio added successfully, length " << length_in_seconds << endl;

    return length_in_seconds;
}

void MovieWriter::add_silence(double duration) {
    cout << "Adding silence: " << duration << " seconds" << endl;

    // Calculate the number of samples needed for the specified duration
    int numSamples = static_cast<int>(duration * audioOutputCodecContext->sample_rate);

    // Calculate the frame size based on the codec context's frame size
    int frameSize = audioOutputCodecContext->frame_size;

    // Calculate the number of frames needed to accommodate the specified duration
    int numFrames = ceil(static_cast<double>(numSamples) / frameSize);

    // Allocate buffer for audio data
    int bufferSize = numSamples * audioOutputCodecContext->channels;
    vector<int16_t> audioBuffer(bufferSize, 0);

    // Split the audio data into multiple frames
    int samplesRemaining = numSamples;

    for (int i = 0; i < numFrames; i++) {
        int samples_this_frame = min(frameSize, samplesRemaining);

        // Fill the audio buffer with silence
        for (int ch = 0; ch < audioOutputCodecContext->channels; ch++) {
            int offset = ch * samples_this_frame;
            for (int s = 0; s < samples_this_frame; s++) {
                audioBuffer[offset + s] = s%5;
            }
        }

        // Create a frame and set its properties
        AVFrame* frame = av_frame_alloc();
        frame->nb_samples = samples_this_frame;
        frame->channel_layout = audioOutputCodecContext->channel_layout;
        frame->sample_rate = audioOutputCodecContext->sample_rate;
        frame->format = audioOutputCodecContext->sample_fmt;

        // Fill the frame with the audio data
        int ret = av_frame_get_buffer(frame, 0);

        for (int ch = 0; ch < audioOutputCodecContext->channels; ch++) {
            frame->linesize[ch] = samples_this_frame * av_get_bytes_per_sample(audioOutputCodecContext->sample_fmt);
        }

        ret = avcodec_fill_audio_frame(frame, audioOutputCodecContext->channels, audioOutputCodecContext->sample_fmt,
                                       reinterpret_cast<const uint8_t*>(audioBuffer.data()), bufferSize, 0);

        frame->pts = audframe;
        audframe++;
        ret = avcodec_send_frame(audioOutputCodecContext, frame);

        encode_and_write_audio();

        av_frame_unref(frame);
        av_frame_free(&frame);

        samplesRemaining -= samples_this_frame;
    }
    cout << "Added silence: " << duration << " seconds" << endl;
}
