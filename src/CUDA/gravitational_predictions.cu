#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>

__device__ float magnitude_force_given_distance_squared_device(float force_constant, float d2) {
    return force_constant / (.1f + d2);
}

__global__ void predict_fate_of_object_kernel(int* planetcolors, glm::vec3* positions, int num_positions, glm::vec3 screen_center, float zoom, int* colors, int* times, int width, int height, const float force_constant, const float collision_threshold_squared, const float drag, const float tick_duration) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    glm::vec3 object_pos(x - width/2, y-height/2, 0);
    object_pos /= zoom;
    object_pos += screen_center;
    glm::vec3 velocity(0.f, 0, 0);
    times[y * width + x] = 0;

    while (times[y * width + x] < 10000) {
        float v2 = glm::dot(velocity, velocity);
        for (int i = 0; i < num_positions; ++i) {
            glm::vec3 direction = positions[i] - object_pos;
            float distance2 = glm::dot(direction, direction);
            if (times[y * width + x] > 5 && distance2 < collision_threshold_squared && v2 < force_constant) {
                colors[y * width + x] = planetcolors[i];
                return;
            } else {
                velocity += tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared_device(force_constant, distance2);
            }
        }

        velocity *= drag;
        object_pos += velocity * tick_duration;
        times[y * width + x]++;
    }
}

extern "C" void render_predictions_cuda(const std::vector<int>& planetcolors, const std::vector<glm::vec3>& positions, int width, int height, glm::vec3 screen_center, float zoom, int* colors, int* times, float force_constant, float collision_threshold_squared, float drag, const float tick_duration) {
    glm::vec3* d_positions;
    int* d_colors;
    int* d_planetcolors;
    int* d_times;
    int num_positions = positions.size();

    cudaMalloc(&d_positions, num_positions * sizeof(glm::vec3));
    cudaMalloc(&d_planetcolors, num_positions * sizeof(int));
    cudaMalloc(&d_colors, width * height * sizeof(int));
    cudaMalloc(&d_times, width * height * sizeof(int));

    cudaMemcpy(d_positions, positions.data(), num_positions * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_planetcolors, planetcolors.data(), num_positions * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    predict_fate_of_object_kernel<<<numBlocks, threadsPerBlock>>>(d_planetcolors, d_positions, num_positions, screen_center, zoom, d_colors, d_times, width, height, force_constant, collision_threshold_squared, drag, tick_duration);

    cudaMemcpy(colors, d_colors, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(times , d_times , width * height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);
    cudaFree(d_planetcolors);
    cudaFree(d_colors);
    cudaFree(d_times);
}

