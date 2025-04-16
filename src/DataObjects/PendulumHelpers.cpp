#ifndef SHARED_PENDULUM_H
#define SHARED_PENDULUM_H

#include <cmath>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

const float pend_g = 9.8f; // gravitational acceleration (m/s^2)
const float pend_l = 1.0f; // length of pendulum arms (m)
const float pend_m = 1.0f; // mass of pendulums (kg)
const float mll = pend_m * pend_l * pend_l;

struct PendulumState {
    float theta1, theta2, p1, p2;
};

struct Derivatives {
    float dtheta1, dtheta2, dp1, dp2;
};

HOST_DEVICE
inline Derivatives computeDerivatives(const PendulumState &state) {
    float theta1 = state.theta1;
    float theta2 = state.theta2;
    float p1 = state.p1;
    float p2 = state.p2;

    float delta = theta1 - theta2;
    float cos_delta = cos(delta);
    float sin_delta = sin(delta);
    float denominator = 16 - 9 * cos_delta * cos_delta;
    float coeff = 6/(mll*denominator);

    float dtheta1 = coeff * (2 * p1 - 3 * cos_delta * p2);
    float dtheta2 = coeff * (8 * p2 - 3 * cos_delta * p1);

    float coeff2 = -0.5f*pend_m*pend_l;
    float endbit = pend_l * dtheta1 * dtheta2 * sin_delta;
    float dp1 = coeff2 * (3 * pend_g * sin(theta1) + endbit);
    float dp2 = coeff2 * (    pend_g * sin(theta2) - endbit);

    return {dtheta1, dtheta2, dp1, dp2};
}

HOST_DEVICE
inline float compute_kinetic_energy(const PendulumState &state) {
    Derivatives d = computeDerivatives(state);
    float cos_delta = cos(state.theta1-state.theta2);
    return mll * (d.dtheta1 * d.dtheta1 + 0.5f * d.dtheta2 * d.dtheta2 + d.dtheta1 * d.dtheta2 * cos_delta);
}

HOST_DEVICE
inline float compute_potential_energy(const PendulumState &state) {
    return pend_m * pend_g * pend_l * (4 - 3 * cos(state.theta1) - cos(state.theta2));
}

HOST_DEVICE
inline PendulumState rk4Step(const PendulumState &state, float dt) {
    Derivatives k1 = computeDerivatives(state);

    PendulumState s2 = {state.theta1 + 0.5f * dt * k1.dtheta1,
        state.theta2 + 0.5f * dt * k1.dtheta2,
        state.p1 + 0.5f * dt * k1.dp1,
        state.p2 + 0.5f * dt * k1.dp2};

    Derivatives k2 = computeDerivatives(s2);

    PendulumState s3 = {state.theta1 + 0.5f * dt * k2.dtheta1,
        state.theta2 + 0.5f * dt * k2.dtheta2,
        state.p1 + 0.5f * dt * k2.dp1,
        state.p2 + 0.5f * dt * k2.dp2};

    Derivatives k3 = computeDerivatives(s3);

    PendulumState s4 = {state.theta1 + dt * k3.dtheta1,
        state.theta2 + dt * k3.dtheta2,
        state.p1 + dt * k3.dp1,
        state.p2 + dt * k3.dp2};

    Derivatives k4 = computeDerivatives(s4);

    PendulumState newState;
    float dt6 = dt/6.0f;
    newState.theta1 = state.theta1 + dt6 * (k1.dtheta1 + 2 * k2.dtheta1 + 2 * k3.dtheta1 + k4.dtheta1);
    newState.theta2 = state.theta2 + dt6 * (k1.dtheta2 + 2 * k2.dtheta2 + 2 * k3.dtheta2 + k4.dtheta2);
    newState.p1 = state.p1 + dt6 * (k1.dp1 + 2 * k2.dp1 + 2 * k3.dp1 + k4.dp1);
    newState.p2 = state.p2 + dt6 * (k1.dp2 + 2 * k2.dp2 + 2 * k3.dp2 + k4.dp2);

    return newState;
}

#endif // SHARED_PENDULUM_H

