#include <cuda_runtime.h>
#include <vector>
#include "../DataObjects/PendulumHelpers.cpp"

// Kernel to update pendulum states
__global__ void pendulum_simulation_kernel(
    PendulumState* states,
    int n,
    int multiplier,
    double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute the next state using the shared RK4 step function
    for(int i = 0; i < multiplier; i++) states[idx] = rk4Step(states[idx], dt);
}

// Host-exposed function to simulate the pendulum
extern "C" void simulatePendulum(
    PendulumState* states, // Pointer to pendulum states in host memory
    int n,                 // Number of pendulums
    int multiplier,
    double dt              // Time step
) {
    PendulumState* d_states;

    // Allocate memory on the device for pendulum states
    cudaMalloc(&d_states, n * sizeof(PendulumState));

    // Copy initial states from host to device
    cudaMemcpy(d_states, states, n * sizeof(PendulumState), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    pendulum_simulation_kernel<<<numBlocks, threadsPerBlock>>>(d_states, n, multiplier, dt);

    // Copy updated states back from device to host
    cudaMemcpy(states, d_states, n * sizeof(PendulumState), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_states);
}

