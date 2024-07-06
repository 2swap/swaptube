#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    void stop_timer(double time) {
        // Stop the timer.
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // Print out time stats
        double render_time_minutes = duration.count() / 60000000.0;
        double video_length_minutes = time/60.;
        cout << "Render time:  " << render_time_minutes << " minutes." << endl;
        cout << "Video length: " << video_length_minutes << " minutes." << endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
