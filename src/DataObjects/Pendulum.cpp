#pragma once

using namespace std;

#include <vector>
#include "DataObject.cpp"
#include "PendulumHelpers.cpp"

extern "C" void simulatePendulum(PendulumState* states, int n, int multiplier, double dt);

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
    vector<PendulumState> pendulum_states;
    vector<PendulumState> pendulum_pairs;
    PendulumGrid(const int width, const int height, const double init_p1 = 0, const double init_p2 = 0) : w(width), h(height) {
        pendulum_states = vector<PendulumState>(h*w);
        pendulum_pairs  = vector<PendulumState>(h*w);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = x + y * w;
                PendulumState ps = {init_p1, init_p2,
                                    //x*2*M_PI/w,
                                    //y*2*M_PI/h,};
                                    x*20./w,
                                    y*20./h,};
                pendulum_states[i] = ps;
                ps.theta1 += 0.001;
                pendulum_pairs [i] = ps;
            }
        }
    }

    void iterate_physics(int multiplier, double step_size) {
        if(multiplier == 0) return;
        simulatePendulum(pendulum_states.data(), pendulum_states.size(), multiplier, step_size);
        simulatePendulum(pendulum_pairs .data(), pendulum_pairs .size(), multiplier, step_size);
        mark_updated();
    }
};
