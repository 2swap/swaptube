#pragma once

bool MovieWriter::encode_and_write_frame(AVFrame* frame){
    int got_output = 0;
    avcodec_encode_video2(videoCodecContext, &pkt, frame, &got_output);
    if (!got_output) return false;

    // We set the packet PTS and DTS taking in the account our FPS (second argument),
    // and the time base that our selected format uses (third argument).
    av_packet_rescale_ts(&pkt, { 1, VIDEO_FRAMERATE }, videoStream->time_base);

    pkt.stream_index = videoStream->index;
    cout << "Writing frame " << outframe << " (size = " << pkt.size << ")" << endl;

    // Write the encoded frame to the mp4 file.
    av_interleaved_write_frame(fc, &pkt);

    return true;
}

void MovieWriter::addFrame(const Pixels& p)
{
    cout << "Encoding frame " << inframe++ << ". ";
    const uint8_t* pixels = &p.pixels[0];

    // The AVFrame data will be stored as RGBRGBRGB... row-wise,
    // from left to right and from top to bottom.
    for (unsigned int y = 0; y < VIDEO_HEIGHT; y++)
    {
        for (unsigned int x = 0; x < VIDEO_WIDTH; x++)
        {
            // rgbpic->linesize[0] is equal to width.
            rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 0] = pixels[y * 4 * VIDEO_WIDTH + 4 * x + 2];
            rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 1] = pixels[y * 4 * VIDEO_WIDTH + 4 * x + 1];
            rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 2] = pixels[y * 4 * VIDEO_WIDTH + 4 * x + 0];
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