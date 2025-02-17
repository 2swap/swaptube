#pragma once

using namespace std;

#include <vector>
#include "DataObject.cpp"
#include "PendulumHelpers.cpp"

extern "C" void simulatePendulum(PendulumState* states, int n, int multiplier, double dt);
extern "C" void simulate_pendulum_pair(PendulumState* states, PendulumState* pairs, double* diffs, int n, int multiplier, double dt);

class Pendulum : public DataObject {
public:
    PendulumState state;
    Pendulum(const PendulumState& s) : state(s) {}

    void iterate_physics(int multiplier, double step_size) {
        for (int step = 0; step < multiplier; ++step) state = rk4Step(state, step_size);
        mark_updated();
    }
};

class PendulumGrid : public DataObject {
public:
    int w; int h;
    double t1_min;
    double t1_max;
    double t2_min;
    double t2_max;
    double p1_min;
    double p1_max;
    double p2_min;
    double p2_max;
    double min1;
    double max1;
    double min2;
    double max2;
    vector<PendulumState> start_states;
    vector<PendulumState> pendulum_states;
    vector<PendulumState> pendulum_pairs;
    vector<double> diff_sums;
    PendulumGrid(const int width, const int height,
        const double t1_min, const double t1_max,
        const double t2_min, const double t2_max,
        const double p1_min, const double p1_max,
        const double p2_min, const double p2_max
    ) : w(width), h(height), t1_min(t1_min), t1_max(t1_max), t2_min(t2_min), t2_max(t2_max), p1_min(p1_min), p1_max(p1_max), p2_min(p2_min), p2_max(p2_max) {
        min1 = (t1_min == t1_max) ? p1_min : t1_min;
        max1 = (t1_min == t1_max) ? p1_max : t1_max;
        min2 = (t2_min == t2_max) ? p2_min : t2_min;
        max2 = (t2_min == t2_max) ? p2_max : t2_max;
        start_states = vector<PendulumState>(h*w);
        pendulum_states = vector<PendulumState>(h*w);
        pendulum_pairs  = vector<PendulumState>(h*w);
        diff_sums       = vector<double       >(h*w);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = x + y * w;
                PendulumState ps = {
                                    lerp(t1_min, t1_max, x/(w-1.)),
                                    lerp(t2_min, t2_max, y/(h-1.)),
                                    lerp(p1_min, p1_max, x/(w-1.)),
                                    lerp(p2_min, p2_max, y/(h-1.)),
                                   };
                pendulum_states[i] = ps;
                start_states[i] = ps;
                ps.theta1 += 0.0001;
                pendulum_pairs[i] = ps;
            }
        }
    }

    void iterate_physics(int multiplier, double step_size) {
        if(multiplier == 0) return;
        simulate_pendulum_pair(pendulum_states.data(), pendulum_pairs.data(), diff_sums.data(), pendulum_states.size(), multiplier, step_size);
        mark_updated();
    }
};
