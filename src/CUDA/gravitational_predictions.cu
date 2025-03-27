#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>

__device__ double magnitude_force_given_distance_squared_device(float eps, float force_constant, float d2) {
    return force_constant / (eps + d2);
}

__global__ void predict_fate_of_object_kernel(
glm::dvec3* positions, const int num_positions, // Planet data
const int width, const int height, const int depth, const glm::dvec3 screen_center, const float zoom, // Geometry of query
const float force_constant, const float collision_threshold_squared, const float drag, const double tick_duration, const float eps, // adjustable parameters
int* colors, int* times // Outputs
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    glm::dvec3 object_pos(x - width/2.f, y - height/2.f, z - depth/2.f);
    object_pos /= zoom;
    object_pos += screen_center;
    glm::dvec3 velocity(0.f, 0, 0);
    const int arr_idx = x + (y + z * height) * width;
    times[arr_idx] = 0;

    while (times[arr_idx] < 30000) {
        float v2 = glm::dot(velocity, velocity);
        for (int i = 0; i < num_positions; ++i) {
            glm::dvec3 direction = positions[i] - object_pos;
            float distance2 = glm::dot(direction, direction);
            if (times[arr_idx] > 5 && distance2 < collision_threshold_squared && v2 < force_constant) {
                colors[arr_idx] = i;
                return;
            } else {
                velocity += tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared_device(eps, force_constant, distance2);
            }
        }

        velocity *= drag;
        object_pos += velocity * tick_duration;
        times[arr_idx]++;
    }
    colors[arr_idx] = -1;
}

extern "C" void render_predictions_cuda(
const std::vector<glm::dvec3>& positions, // Planet data
const int width, const int height, const int depth, const glm::dvec3 screen_center, const float zoom, // Geometry of query
const float force_constant, const float collision_threshold_squared, const float drag, const double tick_duration, const float eps, // Adjustable parameters
int* colors, int* times // outputs
) {
    glm::dvec3* d_positions;
    int* d_colors;
    int* d_times;
    const int num_positions = positions.size();

    cudaMalloc(&d_positions   , num_positions * sizeof(glm::dvec3));
    cudaMalloc(&d_colors, width * height * depth * sizeof(int));
    cudaMalloc(&d_times , width * height * depth * sizeof(int));

    cudaMemcpy(d_positions, positions.data(), num_positions * sizeof(glm::dvec3), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock_thin (16, 16, 4);
    dim3 threadsPerBlock_thick(32, 32, 1);
    dim3 threadsPerBlock = depth>1?threadsPerBlock_thick:threadsPerBlock_thin;
    dim3 numBlocks(( width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   ( depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    predict_fate_of_object_kernel<<<numBlocks, threadsPerBlock>>>(d_positions, num_positions, width, height, depth, screen_center, zoom, force_constant, collision_threshold_squared, drag, tick_duration, eps, d_colors, d_times);

    cudaMemcpy(colors, d_colors, width * height * depth * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(times , d_times , width * height * depth * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);
    cudaFree(d_colors);
    cudaFree(d_times);
}

