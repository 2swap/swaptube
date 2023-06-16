void MovieWriter::add_audio(const string& inputAudioFilename) {
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

void MovieWriter::add_silence(double duration) {
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
