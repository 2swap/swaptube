#pragma once

using namespace std;

#include <vector>
#include "DataObject.h"
#include "../Host_Device_Shared/PendulumHelpers.h"
#include "../Host_Device_Shared/helpers.h"

class Pendulum : public DataObject {
public:
    PendulumState state;
    Pendulum(const PendulumState& s);
    void iterate_physics(int multiplier, pendulum_type step_size);
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
    int samples;
    PendulumGrid(const int width, const int height, const pendulum_type d,
        const pendulum_type t1_min, const pendulum_type t1_max,
        const pendulum_type t2_min, const pendulum_type t2_max,
        const pendulum_type p1_min, const pendulum_type p1_max,
        const pendulum_type p2_min, const pendulum_type p2_max
    );
    void iterate_physics(int multiplier, pendulum_type step_size);
};
