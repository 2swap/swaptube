#include "LivePlayer.h"
#include <iostream>
#include <string>
#include <cstdio>

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

LivePlayer::LivePlayer(const ivec2& dim) : dimensions(dim) {
    string ffplay_cmd_str = "ffplay -f rawvideo -pixel_format argb -video_size " + to_string(dimensions.x) + "x" + to_string(dimensions.y) + " -";
    cout << "Running command: " << ffplay_cmd_str << endl;
    const char* ffplay_cmd = ffplay_cmd_str.c_str();
    pipe = popen(ffplay_cmd, "w");
}

void LivePlayer::accept_frame(uint32_t* device_pixels, bool print) {
cout << "B" << endl;
    Pixels pix(dimensions);
    cuda_copy_pixels_to_host(pix.pixels.data(), pix.pixels.size(), device_pixels);
    if(print) pix.print_to_terminal();
    fwrite(pix.pixels.data(),
        sizeof(int32_t),
        pix.pixels.size(),
        pipe);
    fflush(pipe);
cout << "C" << endl;
}
