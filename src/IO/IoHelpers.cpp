#include "IoHelpers.h"
#include <unistd.h>
#include <cstdio>

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

void restore_stderr(int original_stderr) {
    fflush(stderr);
    dup2(original_stderr, STDERR_FILENO);
    close(original_stderr);
}

std::string read_from_fd(int fd) {
    std::string output;
    char buffer[1024];
    ssize_t bytes_read;

    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        output.append(buffer, bytes_read);
    }

    return output;
}
