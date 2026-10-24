#pragma once

#include <cstdint>

// Common interface for the video encoders, so Writer can hold either the hardware
// GpuVideoWriter (NVENC/VAAPI) or the CpuVideoWriter fallback.
class VideoWriter {
public:
    virtual void add_frame(uint32_t* device_argb_pixels) = 0;
    virtual ~VideoWriter() = default;
};
