#pragma once

#include <unistd.h>
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

// Function to redirect stderr to a pipe
int redirect_stderr(int pipefd[2]) {
    fflush(stderr);
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return -1;
    }

    // Save the original stderr
    int original_stderr = dup(STDERR_FILENO);

    // Redirect stderr to the write end of the pipe
    if (dup2(pipefd[1], STDERR_FILENO) == -1) {
        perror("dup2");
        return -1;
    }

    // Close the write end of the pipe in the parent process
    close(pipefd[1]);

    return original_stderr;
}

// Function to restore the original stderr
void restore_stderr(int original_stderr) {
    fflush(stderr);
    dup2(original_stderr, STDERR_FILENO);
    close(original_stderr);
}

// Function to read from a file descriptor into a string
string read_from_fd(int fd) {
    string output;
    char buffer[1024];
    ssize_t bytes_read;

    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        output.append(buffer, bytes_read);
    }

    return output;
}
