#include "LivePlayer.h"
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

LivePlayer::LivePlayer(const ivec2& dim) : dimensions(dim) {
    // Quiet, so that ffplay's status line doesn't clobber the prompt.
    const string ffplay_cmd = "ffplay -loglevel error -nostats -autoexit -f rawvideo -pixel_format bgra -video_size " + to_string(dimensions.x) + "x" + to_string(dimensions.y) + " -";
    cout << "Running command: " << ffplay_cmd << endl;
    pipe = popen(ffplay_cmd.c_str(), "w");
    if (!pipe) throw runtime_error("LivePlayer: failed to run ffplay");

    // Writing to a player whose window was closed must not kill the render.
    old_sigpipe_handler = signal(SIGPIPE, SIG_IGN);
}

LivePlayer::~LivePlayer() {
    pclose(pipe);
    signal(SIGPIPE, old_sigpipe_handler);
}

bool LivePlayer::accept_frame(uint32_t* device_pixels, bool print) {
    Pixels pix(dimensions);
    cuda_copy_pixels_to_host(pix.pixels.data(), pix.pixels.size(), device_pixels);
    if(print) pix.print_to_terminal();
    const size_t written = fwrite(pix.pixels.data(), sizeof(int32_t), pix.pixels.size(), pipe);
    return written == pix.pixels.size() && fflush(pipe) == 0;
}
