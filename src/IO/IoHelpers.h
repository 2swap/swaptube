#pragma once

#include <string>

extern "C" {
    #include <libavutil/error.h>  // AV_ERROR_MAX_STRING_SIZE, av_strerror
}

static inline std::string ff_errstr(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    if (av_strerror(errnum, buf, sizeof(buf)) < 0) {
        return "Unknown FFmpeg error " + std::to_string(errnum);
    }
    return std::string(buf);
}

// Function declarations
int redirect_stderr(int pipefd[2]);
void restore_stderr(int original_stderr);
std::string read_from_fd(int fd);
