#include <chrono>

class Timer {
public:
    Timer();
    ~Timer();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
