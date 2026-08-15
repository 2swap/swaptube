#pragma once
#include <string>
#include "../Host_Device_Shared/vec.h"
#include "../Core/Pixels.h"

class LivePlayer {
public:
    LivePlayer(const ivec2& dimensions);
    void accept_frame(uint32_t* device_pixels, bool print);
private:
    FILE* pipe;
    ivec2 dimensions;
};
