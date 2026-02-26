#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/PendulumHelpers.h"

// Kernel to update pendulum states
__global__ void pendulum_simulation_kernel(
    Cuda::PendulumState* d_states,
    int n,
    int multiplier,
    Cuda::pendulum_type dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute the next state using the shared RK4 step function
    for(int i = 0; i < multiplier; i++) d_states[idx] = rk4Step(d_states[idx], dt);
}

// Host-exposed function to simulate the pendulum
extern "C" void simulatePendulum(
    Cuda::PendulumState* states, // Pointer to pendulum states in host memory
    int n,                 // Number of pendulums
    int multiplier,
    Cuda::pendulum_type dt              // Time step
) {
    Cuda::PendulumState* d_states;

    // Allocate memory on the device for pendulum states
    cudaMalloc(&d_states, n * sizeof(Cuda::PendulumState));

    // Copy initial states from host to device
    cudaMemcpy(d_states, states, n * sizeof(Cuda::PendulumState), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    pendulum_simulation_kernel<<<numBlocks, threadsPerBlock>>>(d_states, n, multiplier, dt);

    // Copy updated states back from device to host
    cudaMemcpy(states, d_states, n * sizeof(Cuda::PendulumState), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_states);
}

// Kernel to update pendulum states
__global__ void double_pendulum_simulation_kernel(
    Cuda::PendulumState* d_states,
    Cuda::PendulumState* d_pairs,
    Cuda::pendulum_type* d_diffs,
    int n,
    int multiplier,
    Cuda::pendulum_type dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute the next state using the shared RK4 step function
    for(int i = 0; i < multiplier; i++) {
        d_states[idx] = rk4Step(d_states[idx], dt);
        d_pairs[idx] = rk4Step(d_pairs[idx], dt);
        Cuda::pendulum_type p1_dist = d_states[idx].p1 - d_pairs[idx].p1;
        Cuda::pendulum_type p2_dist = d_states[idx].p2 - d_pairs[idx].p2;
        Cuda::pendulum_type theta1_dist = d_states[idx].theta1 - d_pairs[idx].theta1;
        Cuda::pendulum_type theta2_dist = d_states[idx].theta2 - d_pairs[idx].theta2;
        Cuda::pendulum_type distance = sqrt(p1_dist*p1_dist + p2_dist*p2_dist + theta1_dist*theta1_dist + theta2_dist*theta2_dist);
        distance = min(distance, 1.f);
        d_diffs[idx] += distance;
    }
}

// Host-exposed function to simulate the pendulum
extern "C" void simulate_pendulum_pair(
    Cuda::PendulumState* states, // Pointer to pendulum states in host memory
    Cuda::PendulumState* pairs, // Pointer to pendulum states in host memory
    Cuda::pendulum_type* diffs, // Pointer to pendulum states in host memory
    int n,                 // Number of pendulums
    int multiplier,
    Cuda::pendulum_type dt              // Time step
) {
    Cuda::PendulumState* d_states;
    Cuda::PendulumState* d_pairs;
    Cuda::pendulum_type* d_diffs;

    // Allocate memory on the device for pendulum states
    cudaMalloc(&d_states, n * sizeof(Cuda::PendulumState));
    cudaMalloc(&d_pairs , n * sizeof(Cuda::PendulumState));
    cudaMalloc(&d_diffs , n * sizeof(Cuda::pendulum_type));

    // Copy initial states from host to device
    cudaMemcpy(d_states, states, n * sizeof(Cuda::PendulumState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs , pairs , n * sizeof(Cuda::PendulumState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_diffs , diffs , n * sizeof(Cuda::pendulum_type), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    double_pendulum_simulation_kernel<<<numBlocks, threadsPerBlock>>>(d_states, d_pairs, d_diffs, n, multiplier, dt);

    // Copy updated states back from device to host
    cudaMemcpy(states, d_states, n * sizeof(Cuda::PendulumState), cudaMemcpyDeviceToHost);
    cudaMemcpy(pairs , d_pairs , n * sizeof(Cuda::PendulumState), cudaMemcpyDeviceToHost);
    cudaMemcpy(diffs , d_diffs , n * sizeof(Cuda::pendulum_type), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_states);
    cudaFree(d_pairs);
    cudaFree(d_diffs);
}

