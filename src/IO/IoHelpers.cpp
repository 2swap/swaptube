#include "IoHelpers.h"
#include <cstdio>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

int redirect_stderr(int pipefd[2]) {
    fflush(stderr);
#ifdef _WIN32
    if (_pipe(pipefd, 4096, _O_BINARY) == -1) {
#else
    if (pipe(pipefd) == -1) {
#endif
        perror("pipe");
        return -1;
    }

    // Save the original stderr
#ifdef _WIN32
    int original_stderr = _dup(_fileno(stderr));
#else
    int original_stderr = dup(STDERR_FILENO);
#endif

    // Redirect stderr to the write end of the pipe
#ifdef _WIN32
    if (_dup2(pipefd[1], _fileno(stderr)) == -1) {
#else
    if (dup2(pipefd[1], STDERR_FILENO) == -1) {
#endif
        perror("dup2");
        return -1;
    }

    // Close the write end of the pipe in the parent process
    close_fd_portable(pipefd[1]);

    return original_stderr;
}

void restore_stderr(int original_stderr) {
    fflush(stderr);
#ifdef _WIN32
    _dup2(original_stderr, _fileno(stderr));
#else
    dup2(original_stderr, STDERR_FILENO);
#endif
    close_fd_portable(original_stderr);
}

std::string read_from_fd(int fd) {
    std::string output;
    char buffer[1024];
#ifdef _WIN32
    int bytes_read;
    while ((bytes_read = _read(fd, buffer, static_cast<unsigned int>(sizeof(buffer)))) > 0) {
#else
    ssize_t bytes_read;
    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
#endif
        output.append(buffer, bytes_read);
    }

    return output;
}

int close_fd_portable(int fd) {
#ifdef _WIN32
    return _close(fd);
#else
    return close(fd);
#endif
}

FILE* portable_popen(const char* command, const char* mode) {
#ifdef _WIN32
    return _popen(command, mode);
#else
    return popen(command, mode);
#endif
}

int portable_pclose(FILE* pipe) {
#ifdef _WIN32
    return _pclose(pipe);
#else
    return pclose(pipe);
#endif
}
