#pragma once

using namespace std;

#include <vector>
#include "DataObject.cpp"
#include "PendulumHelpers.cpp"

extern "C" void simulatePendulum(PendulumState* states, int n, int multiplier, float dt);
extern "C" void simulate_pendulum_pair(PendulumState* states, PendulumState* pairs, float* diffs, int n, int multiplier, float dt);

class Pendulum : public DataObject {
public:
    PendulumState state;
    Pendulum(const PendulumState& s) : state(s) {}

    void iterate_physics(int multiplier, float step_size) {
        for (int step = 0; step < multiplier; ++step) state = rk4Step(state, step_size);
        mark_updated();
    }
};

class PendulumGrid : public DataObject {
public:
    int w; int h;
    float delta;
    float t1_min;
    float t1_max;
    float t2_min;
    float t2_max;
    float p1_min;
    float p1_max;
    float p2_min;
    float p2_max;
    float min1;
    float max1;
    float min2;
    float max2;
    vector<PendulumState> start_states;
    vector<PendulumState> pendulum_states;
    vector<PendulumState> pendulum_pairs;
    vector<float> diff_sums;
    int samples = 0;
    PendulumGrid(const int width, const int height, const float d,
        const float t1_min, const float t1_max,
        const float t2_min, const float t2_max,
        const float p1_min, const float p1_max,
        const float p2_min, const float p2_max
    ) : w(width), h(height), delta(d), t1_min(t1_min), t1_max(t1_max), t2_min(t2_min), t2_max(t2_max), p1_min(p1_min), p1_max(p1_max), p2_min(p2_min), p2_max(p2_max) {
        min1 = (t1_min == t1_max) ? p1_min : t1_min;
        max1 = (t1_min == t1_max) ? p1_max : t1_max;
        min2 = (t2_min == t2_max) ? p2_min : t2_min;
        max2 = (t2_min == t2_max) ? p2_max : t2_max;
        start_states = vector<PendulumState>(h*w);
        pendulum_states = vector<PendulumState>(h*w);
        pendulum_pairs  = vector<PendulumState>(h*w);
        diff_sums       = vector<float        >(h*w);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = x + y * w;
                PendulumState ps = {
                                    float_lerp(t1_min, t1_max, x/(w-1.f)),
                                    float_lerp(t2_min, t2_max, y/(h-1.f)),
                                    float_lerp(p1_min, p1_max, x/(w-1.f)),
                                    float_lerp(p2_min, p2_max, y/(h-1.f)),
                                   };
                pendulum_states[i] = ps;
                start_states[i] = ps;
                ps.theta1 += delta;
                pendulum_pairs[i] = ps;
            }
        }
    }

    void iterate_physics(int multiplier, float step_size) {
        if(multiplier == 0) return;
        simulate_pendulum_pair(pendulum_states.data(), pendulum_pairs.data(), diff_sums.data(), pendulum_states.size(), multiplier, step_size);
        samples += multiplier;
        mark_updated();
    }
};
