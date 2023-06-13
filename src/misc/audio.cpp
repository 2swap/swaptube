

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

        AVCodec* audioDecoder = avcodec_find_decoder(inputFormatContext->streams[audioStreamIndex]->codecpar->codec_id);
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

        ret = avcodec_parameters_to_context(audioDecoderContext, inputFormatContext->streams[audioStreamIndex]->codecpar);
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
