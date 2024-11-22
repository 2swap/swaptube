#include <cuda_runtime.h>
#include <glm/glm.hpp>

__global__ void compute_repulsion_kernel(const glm::dvec4* positions, glm::dvec4* velocity_deltas, int num_nodes, double repel_force) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_nodes) return;

    glm::dvec4 pos_i = positions[i];
    glm::dvec4 delta = glm::dvec4(0.0);

    for (int j = 0; j < num_nodes; ++j) {
        if (i == j) continue; // Skip self-interaction

        glm::dvec4 pos_j = positions[j];
        glm::dvec4 diff = pos_i - pos_j;
        double dist_sq = glm::dot(diff, diff) + 1; // Add a small epsilon to prevent division by zero
        double force = -repel_force / (dist_sq / 2.0 + 0.1); // Repulsion force formula

        delta -= diff * force;
    }

    velocity_deltas[i] = delta;
}

extern "C" void compute_repulsion_cuda(const glm::dvec4* host_positions, glm::dvec4* host_velocity_deltas, int num_nodes, double repel_force) {
    glm::dvec4 *d_positions, *d_velocity_deltas;

    size_t size = num_nodes * sizeof(glm::dvec4);

    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_velocity_deltas, size);

    cudaMemcpy(d_positions, host_positions, size, cudaMemcpyHostToDevice);
    cudaMemset(d_velocity_deltas, 0, size);

    int blockSize = 256;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;

    compute_repulsion_kernel<<<gridSize, blockSize>>>(d_positions, d_velocity_deltas, num_nodes, repel_force);

    cudaMemcpy(host_velocity_deltas, d_velocity_deltas, size, cudaMemcpyDeviceToHost);

    cudaFree(d_positions);
    cudaFree(d_velocity_deltas);
}
