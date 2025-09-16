#pragma once

extern "C" {
#include <libavutil/error.h>  // AV_ERROR_MAX_STRING_SIZE, av_strerror
}

#include <string>

static inline std::string ff_errstr(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    if (av_strerror(errnum, buf, sizeof(buf)) < 0) {
        return "Unknown FFmpeg error " + std::to_string(errnum);
    }
    return std::string(buf);
}

