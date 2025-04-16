#pragma once

using namespace std;

#include <vector>
#include "DataObject.cpp"
#include "PendulumHelpers.cpp"

extern "C" void simulatePendulum(PendulumState* states, int n, int multiplier, pendulum_type dt);
extern "C" void simulate_pendulum_pair(PendulumState* states, PendulumState* pairs, pendulum_type* diffs, int n, int multiplier, pendulum_type dt);

class Pendulum : public DataObject {
public:
    PendulumState state;
    Pendulum(const PendulumState& s) : state(s) {}

    void iterate_physics(int multiplier, pendulum_type step_size) {
        for (int step = 0; step < multiplier; ++step) state = rk4Step(state, step_size);
        mark_updated();
    }
};

class PendulumGrid : public DataObject {
public:
    int w; int h;
    pendulum_type delta;
    pendulum_type t1_min;
    pendulum_type t1_max;
    pendulum_type t2_min;
    pendulum_type t2_max;
    pendulum_type p1_min;
    pendulum_type p1_max;
    pendulum_type p2_min;
    pendulum_type p2_max;
    pendulum_type min1;
    pendulum_type max1;
    pendulum_type min2;
    pendulum_type max2;
    vector<PendulumState> start_states;
    vector<PendulumState> pendulum_states;
    vector<PendulumState> pendulum_pairs;
    vector<pendulum_type> diff_sums;
    int samples = 0;
    PendulumGrid(const int width, const int height, const pendulum_type d,
        const pendulum_type t1_min, const pendulum_type t1_max,
        const pendulum_type t2_min, const pendulum_type t2_max,
        const pendulum_type p1_min, const pendulum_type p1_max,
        const pendulum_type p2_min, const pendulum_type p2_max
    ) : w(width), h(height), delta(d), t1_min(t1_min), t1_max(t1_max), t2_min(t2_min), t2_max(t2_max), p1_min(p1_min), p1_max(p1_max), p2_min(p2_min), p2_max(p2_max) {
        min1 = (t1_min == t1_max) ? p1_min : t1_min;
        max1 = (t1_min == t1_max) ? p1_max : t1_max;
        min2 = (t2_min == t2_max) ? p2_min : t2_min;
        max2 = (t2_min == t2_max) ? p2_max : t2_max;
        start_states = vector<PendulumState>(h*w);
        pendulum_states = vector<PendulumState>(h*w);
        pendulum_pairs  = vector<PendulumState>(h*w);
        diff_sums       = vector<pendulum_type>(h*w);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = x + y * w;
                PendulumState ps = {
                                    static_cast<pendulum_type>(lerp(t1_min, t1_max, x/(w-1.))),
                                    static_cast<pendulum_type>(lerp(t2_min, t2_max, y/(h-1.))),
                                    static_cast<pendulum_type>(lerp(p1_min, p1_max, x/(w-1.))),
                                    static_cast<pendulum_type>(lerp(p2_min, p2_max, y/(h-1.))),
                                   };
                pendulum_states[i] = ps;
                start_states[i] = ps;
                ps.theta1 += delta;
                pendulum_pairs[i] = ps;
            }
        }
    }

    void iterate_physics(int multiplier, pendulum_type step_size) {
        if(multiplier == 0) return;
        simulate_pendulum_pair(pendulum_states.data(), pendulum_pairs.data(), diff_sums.data(), pendulum_states.size(), multiplier, step_size);
        samples += multiplier;
        mark_updated();
    }
};
