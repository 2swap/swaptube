#pragma once

using namespace std;

#include <vector>
#include "DataObject.cpp"

const double g = 9.8; // gravitational acceleration (m/s^2)
const double l = 1.0; // length of pendulum arms (m)
const double m = 1.0; // mass of pendulums (kg)

struct PendulumState {
    double theta1, theta2, p1, p2;
};

struct Derivatives {
    double dtheta1, dtheta2, dp1, dp2;
};

Derivatives computeDerivatives(const PendulumState &state) {
    double theta1 = state.theta1;
    double theta2 = state.theta2;
    double p1 = state.p1;
    double p2 = state.p2;

    double delta = theta1 - theta2;
    double cos_delta = cos(delta);
    double sin_delta = sin(delta);
    double denominator = 16 - 9 * cos_delta * cos_delta;
    double coeff = 6/(m*l*l*denominator);

    double dtheta1 = coeff * (2 * p1 - 3 * cos_delta * p2);
    double dtheta2 = coeff * (8 * p2 - 3 * cos_delta * p1);

    double coeff2 = -0.5*m*l;
    double endbit = l * dtheta1 * dtheta2 * sin_delta;
    double dp1 = coeff2 * (3 * g * sin(theta1) + endbit);
    double dp2 = coeff2 * (    g * sin(theta2) - endbit);

    return {dtheta1, dtheta2, dp1, dp2};
}

PendulumState rk4Step(const PendulumState &state, double dt) {
    Derivatives k1 = computeDerivatives(state);

    PendulumState s2 = {state.theta1 + 0.5 * dt * k1.dtheta1,
	state.theta2 + 0.5 * dt * k1.dtheta2,
	state.p1 + 0.5 * dt * k1.dp1,
	state.p2 + 0.5 * dt * k1.dp2};

    Derivatives k2 = computeDerivatives(s2);

    PendulumState s3 = {state.theta1 + 0.5 * dt * k2.dtheta1,
	state.theta2 + 0.5 * dt * k2.dtheta2,
	state.p1 + 0.5 * dt * k2.dp1,
	state.p2 + 0.5 * dt * k2.dp2};

    Derivatives k3 = computeDerivatives(s3);

    PendulumState s4 = {state.theta1 + dt * k3.dtheta1,
        state.theta2 + dt * k3.dtheta2,
        state.p1 + dt * k3.dp1,
        state.p2 + dt * k3.dp2};

    Derivatives k4 = computeDerivatives(s4);

    PendulumState newState;
    newState.theta1 = state.theta1 + (dt / 6.0) * (k1.dtheta1 + 2 * k2.dtheta1 + 2 * k3.dtheta1 + k4.dtheta1);
    newState.theta2 = state.theta2 + (dt / 6.0) * (k1.dtheta2 + 2 * k2.dtheta2 + 2 * k3.dtheta2 + k4.dtheta2);
    newState.p1 = state.p1 + (dt / 6.0) * (k1.dp1 + 2 * k2.dp1 + 2 * k3.dp1 + k4.dp1);
    newState.p2 = state.p2 + (dt / 6.0) * (k1.dp2 + 2 * k2.dp2 + 2 * k3.dp2 + k4.dp2);

    return newState;
}

class Pendulum : public DataObject {
public:
    PendulumState state;
    Pendulum(const PendulumState& s) : state(s) {}

    void iterate_physics(int multiplier, double step_size) {
        for (int step = 0; step < multiplier; ++step) state = rk4Step(state, step_size);
        mark_updated();
    }
};

