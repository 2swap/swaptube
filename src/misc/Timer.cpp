#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    ~Timer(){
        // Stop the timer.
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // Print out time stats
        double render_time_minutes = duration.count() / 60000000.0;
        cout << "=================== Timer Output ===================" << endl;
        cout << "= Render time:  " << render_time_minutes << " minutes." << endl;
        cout << "====================================================" << endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
