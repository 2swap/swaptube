#ifndef SHARED_PENDULUM_H
#define SHARED_PENDULUM_H

#include <cmath>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

const double pend_g = 9.8; // gravitational acceleration (m/s^2)
const double pend_l = 1.0; // length of pendulum arms (m)
const double pend_m = 1.0; // mass of pendulums (kg)

struct PendulumState {
    double theta1, theta2, p1, p2;
    int iteration_number;
};

struct Derivatives {
    double dtheta1, dtheta2, dp1, dp2;
};

HOST_DEVICE
inline double compute_kinetic_energy(const PendulumState &state) {
    double theta1 = state.theta1;
    double theta2 = state.theta2;
    double p1 = state.p1;
    double p2 = state.p2;

    // Calculate angular velocities
    double delta = theta1 - theta2;
    double denominator = 16 - 9 * cos(delta) * cos(delta);
    double mll = pend_m * pend_l * pend_l;
    double dtheta1 = (6 / mll) * (2 * p1 - 3 * cos(delta) * p2) / denominator;
    double dtheta2 = (6 / mll) * (8 * p2 - 3 * cos(delta) * p1) / denominator;

    // Kinetic Energy
    return 0.5 * mll * (dtheta1 * dtheta1 + 2 * dtheta2 * dtheta2 + 2 * dtheta1 * dtheta2 * cos(delta));
}

HOST_DEVICE
inline double compute_potential_energy(const PendulumState &state) {
    return pend_m * pend_g * pend_l * (4 - 3 * cos(state.theta1) - cos(state.theta2));
}

HOST_DEVICE
inline Derivatives computeDerivatives(const PendulumState &state) {
    double theta1 = state.theta1;
    double theta2 = state.theta2;
    double p1 = state.p1;
    double p2 = state.p2;

    double delta = theta1 - theta2;
    double cos_delta = cos(delta);
    double sin_delta = sin(delta);
    double denominator = 16 - 9 * cos_delta * cos_delta;
    double coeff = 6/(pend_m*pend_l*pend_l*denominator);

    double dtheta1 = coeff * (2 * p1 - 3 * cos_delta * p2);
    double dtheta2 = coeff * (8 * p2 - 3 * cos_delta * p1);

    double coeff2 = -0.5*pend_m*pend_l;
    double endbit = pend_l * dtheta1 * dtheta2 * sin_delta;
    double dp1 = coeff2 * (3 * pend_g * sin(theta1) + endbit);
    double dp2 = coeff2 * (    pend_g * sin(theta2) - endbit);

    return {dtheta1, dtheta2, dp1, dp2};
}

HOST_DEVICE
inline PendulumState rk4Step(const PendulumState &state, double dt) {
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
    double dt6 = dt/6.0;
    newState.theta1 = state.theta1 + dt6 * (k1.dtheta1 + 2 * k2.dtheta1 + 2 * k3.dtheta1 + k4.dtheta1);
    newState.theta2 = state.theta2 + dt6 * (k1.dtheta2 + 2 * k2.dtheta2 + 2 * k3.dtheta2 + k4.dtheta2);
    newState.p1 = state.p1 + dt6 * (k1.dp1 + 2 * k2.dp1 + 2 * k3.dp1 + k4.dp1);
    newState.p2 = state.p2 + dt6 * (k1.dp2 + 2 * k2.dp2 + 2 * k3.dp2 + k4.dp2);
    newState.iteration_number = state.iteration_number+1;

    return newState;
}

#endif // SHARED_PENDULUM_H

