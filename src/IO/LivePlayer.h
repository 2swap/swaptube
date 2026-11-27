#pragma once
#include <cstdio>
#include <string>
#include "../Host_Device_Shared/vec.h"
#include "../Core/Pixels.h"

// Shows frames in an ffplay window.
class LivePlayer {
public:
    LivePlayer(const ivec2& dimensions);
    ~LivePlayer();
    LivePlayer(const LivePlayer&) = delete;
    LivePlayer& operator=(const LivePlayer&) = delete;

    // Returns false if the player is no longer running (e.g. its window was closed).
    bool accept_frame(uint32_t* device_pixels, bool print);
private:
    FILE* pipe;
    ivec2 dimensions;
    void (*old_sigpipe_handler)(int);
};
