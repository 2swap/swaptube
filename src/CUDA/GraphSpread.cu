#include <cuda_runtime.h>
#include <stdio.h>
#include <glm/glm.hpp>

#define GRID_SIZE 10 // 10x10x10 bins
#define BIN_INDEX(a) ((a.z) * GRID_SIZE * GRID_SIZE + (a.y) * GRID_SIZE + (a.x))

struct Bin {
    int count;
    glm::vec4 center_of_mass;
};

__device__ float clamp(float min, float val, float max) {
    return val < min? min : (val > max? max : val);
}

__device__ glm::vec4 compute_force(glm::vec4 pos_i, glm::vec4 pos_j) {
    glm::vec4 diff = pos_i - pos_j;
    float dist_sq = glm::dot(diff, diff) + 1.0f;
    glm::vec4 norm = glm::normalize(diff);
    glm::vec4 result = norm / (dist_sq * 10.0f + 2.0f);
    return result;
}

__global__ void compute_repulsion_kernel_naive(const glm::vec4* positions, glm::vec4* velocities, int num_nodes, float repel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    glm::vec4 pos_i = positions[i];
    glm::vec4 delta(0.0f);

    for (int j = 0; j < num_nodes; ++j) {
        if (i == j) continue;
        delta += compute_force(pos_i, positions[j]);
    }

    velocities[i] += delta * repel;
}


void sort_positions_by_bins_with_indices(const glm::vec4* positions, glm::vec4* sorted_positions, 
                                         const int* node_bins, int* sorted_indices, int num_nodes, int* bin_counts) {
    // Step 1: Compute cumulative bin counts to determine sorted indices
    int num_bins = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    int* bin_offsets = new int[num_bins];
    bin_offsets[0] = 0;

    for (int i = 0; i < num_bins - 1; ++i) {
        bin_offsets[i + 1] = bin_offsets[i] + bin_counts[i];
    }

    // Step 2: Insert nodes into their sorted positions
    for (int i = 0; i < num_nodes; ++i) {
        int bin_idx = node_bins[i];
        int sorted_idx = bin_offsets[bin_idx]; // Get the current position and increment the offset

        sorted_positions[sorted_idx] = positions[i];
        sorted_indices[sorted_idx] = i;

        bin_offsets[bin_idx]++;
    }

    // Cleanup
    delete[] bin_offsets;
}

__global__ void compute_repulsion_kernel_binned(const glm::vec4* sorted_positions, glm::vec4* velocities,
                                                const Bin* bins, const int* bin_start_indices, 
                                                const int* sorted_indices, int num_nodes, 
                                                glm::vec4 min_bounds, glm::vec4 bin_size, float repel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    glm::vec4 pos_i = sorted_positions[i];
    glm::vec4 delta = glm::vec4(0.0f);

    // Determine the bin index for the current node
    glm::ivec3 bin_idx = glm::ivec3((pos_i - min_bounds) / bin_size);
    bin_idx = glm::clamp(bin_idx, glm::ivec3(0), glm::ivec3(GRID_SIZE - 1));

    // Step 1: Interact with nodes in neighboring bins
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                glm::ivec3 neighbor_bin_idx = bin_idx + glm::ivec3(dx, dy, dz);

                // Skip bins outside valid range
                if (neighbor_bin_idx.x < 0 || neighbor_bin_idx.x >= GRID_SIZE ||
                    neighbor_bin_idx.y < 0 || neighbor_bin_idx.y >= GRID_SIZE ||
                    neighbor_bin_idx.z < 0 || neighbor_bin_idx.z >= GRID_SIZE) {
                    continue;
                }

                int neighbor_bin_flat_idx = BIN_INDEX(neighbor_bin_idx);
                int start_idx = bin_start_indices[neighbor_bin_flat_idx];
                int end_idx = start_idx + bins[neighbor_bin_flat_idx].count;

                for (int j = start_idx; j < end_idx; ++j) {
                    if (sorted_indices[i] == sorted_indices[j]) continue; // Skip self-interaction

                    delta += compute_force(pos_i, sorted_positions[j]);
                }
            }
        }
    }

    // Step 2: Interact with non-neighboring bins (bin-level interaction)
    for (int z = 0; z < GRID_SIZE; ++z) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                glm::ivec3 current_bin_idx_3d(x, y, z);

                // Skip the 3x3x3 neighborhood
                if (abs(current_bin_idx_3d.x - bin_idx.x) <= 1 &&
                    abs(current_bin_idx_3d.y - bin_idx.y) <= 1 &&
                    abs(current_bin_idx_3d.z - bin_idx.z) <= 1) {
                    continue;
                }

                int non_neighbor_bin_idx = BIN_INDEX(current_bin_idx_3d);
                Bin bin = bins[non_neighbor_bin_idx];

                if (bin.count == 0) continue; // Skip empty bins

                delta += (float)bin.count * compute_force(pos_i, bin.center_of_mass);
            }
        }
    }

    velocities[sorted_indices[i]] += delta * repel;
}

__device__ float atomicMin_float(float* address, float val) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val <= val) break;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(old);
}

__device__ float atomicMax_float(float* address, float val) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val >= val) break;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(old);
}

__global__ void populate_bins(const glm::vec4* positions, Bin* bins, int num_nodes, glm::vec4 min_bounds, glm::vec4 bin_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    glm::vec4 pos = positions[i];
    glm::ivec3 bin_idx = glm::ivec3((pos - min_bounds) / bin_size);
    bin_idx = glm::clamp(bin_idx, glm::ivec3(0), glm::ivec3(GRID_SIZE - 1));

    int bin_flat_idx = BIN_INDEX(bin_idx);

    // Atomic operations to update the bin's data
    atomicAdd(&bins[bin_flat_idx].count, 1);
    atomicAdd(&bins[bin_flat_idx].center_of_mass.x, pos.x);
    atomicAdd(&bins[bin_flat_idx].center_of_mass.y, pos.y);
    atomicAdd(&bins[bin_flat_idx].center_of_mass.z, pos.z);
    atomicAdd(&bins[bin_flat_idx].center_of_mass.w, pos.w);
}

__global__ void finalize_bins(Bin* bins, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bins) return;

    if (bins[i].count > 0) {
        bins[i].center_of_mass /= float(bins[i].count);
    }
}

__global__ void compute_aabb(const glm::vec4* positions, int num_nodes, glm::vec4* min_bounds, glm::vec4* max_bounds) {
    __shared__ glm::vec4 local_min;
    __shared__ glm::vec4 local_max;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        local_min = glm::vec4(FLT_MAX);
        local_max = glm::vec4(-FLT_MAX);
    }
    __syncthreads();

    // Update local min/max
    if (i < num_nodes) {
        atomicMin_float(&local_min.x, positions[i].x);
        atomicMin_float(&local_min.y, positions[i].y);
        atomicMin_float(&local_min.z, positions[i].z);

        atomicMax_float(&local_max.x, positions[i].x);
        atomicMax_float(&local_max.y, positions[i].y);
        atomicMax_float(&local_max.z, positions[i].z);
    }
    __syncthreads();

    // Update global min/max using shared memory
    if (tid == 0) {
        atomicMin_float(&min_bounds->x, local_min.x);
        atomicMin_float(&min_bounds->y, local_min.y);
        atomicMin_float(&min_bounds->z, local_min.z);

        atomicMax_float(&max_bounds->x, local_max.x);
        atomicMax_float(&max_bounds->y, local_max.y);
        atomicMax_float(&max_bounds->z, local_max.z);
    }
}

void compute_node_bins(const glm::vec4* positions, int* node_bins, int num_nodes, glm::vec4 min_bounds, glm::vec4 bin_size) {
    for (int i = 0; i < num_nodes; ++i) {
        glm::ivec3 bin_idx = glm::ivec3((positions[i] - min_bounds) / bin_size);
        bin_idx = glm::clamp(bin_idx, glm::ivec3(0), glm::ivec3(GRID_SIZE - 1));
        node_bins[i] = BIN_INDEX(bin_idx);
    }
}

// Compute attraction force helper function on device
// Return float attraction scalar
__device__ float get_attraction_force (float dist_sq) {
    float dist_6th = dist_sq * dist_sq * dist_sq * .05f;
    return (dist_6th - 1.0f) / (dist_6th + 1.0f) * .2f - .1f;
};

// Kernel to compute attraction forces
// Each thread corresponds to one node
__global__ void attraction_kernel(const glm::vec4* positions, const int* adjacency_matrix, glm::vec4* velocities, int num_nodes, int max_degree, float attract) {
    if (adjacency_matrix != nullptr && max_degree > 0) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_nodes) return;

        glm::vec4 pos_i = positions[i];
        glm::vec4 delta(0.0f);

        for (int k = 0; k < max_degree; ++k) {
            int neighbor_idx = adjacency_matrix[i * max_degree + k];
            if (neighbor_idx < 0 || neighbor_idx >= num_nodes) {
                break;
            }
            glm::vec4 pos_j = positions[neighbor_idx];
            glm::vec4 diff = pos_i - pos_j;
            float dist_sq = glm::dot(diff, diff) + 1.0f;

            delta += diff * get_attraction_force(dist_sq);
        }

        velocities[i] -= attract * delta;
    }
}

// Kernel to compute mirror forces
__global__ void mirror_kernel(glm::vec4* positions, glm::vec4* velocities, const int* mirrors, const int* mirror2s, int num_nodes, float mirror_force, const float decay, const float dimension) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    if (mirror_force > 0.001) {
        glm::vec4 pos_i = positions[i];
        glm::vec4 delta(0.0f);

        {
            glm::vec4 mirror = positions[mirrors[i]];
            mirror.x *= -1;
            delta += mirror - pos_i;
        }

        {
            glm::vec4 mirror = positions[mirror2s[i]];
            mirror.y *= -1;
            delta += mirror - pos_i;
        }

        delta.w = 0;
        velocities[i] += mirror_force * delta;
    }

    if(glm::any(glm::isnan(velocities[i]))) { velocities[i] = glm::vec4(0, 0, 0, 0); }
    if(glm::any(glm::isnan( positions[i]))) {  positions[i] = glm::vec4(0, 0, 0, 0); }
    velocities[i] *= decay;
    positions[i] += velocities[i];
    positions[i].z *= clamp(0, dimension - 2, 1);
    positions[i].w *= clamp(0, dimension - 3, 1);
}

extern "C" void compute_repulsion_cuda(glm::vec4* h_positions, glm::vec4* h_velocities, const int* h_adjacency_matrix, const int* h_mirrors, const int* h_mirror2s, int num_nodes, int max_degree, float attract, float repel, float mirror_force, const float decay, const float dimension, const int iterations) {
    if(num_nodes < 0) return;

    glm::vec4 *d_positions;
    glm::vec4 *d_velocities;
    size_t vec4_size = num_nodes * sizeof(glm::vec4);
    cudaMalloc(&d_velocities, vec4_size);
    cudaMemcpy(d_velocities, h_velocities, vec4_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_positions, vec4_size);
    cudaMemcpy(d_positions, h_positions, vec4_size, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;

    int* d_mirrors;
    int* d_mirror2s;
    size_t mirrors_size = num_nodes * sizeof(int);

    cudaMalloc(&d_mirrors, mirrors_size);
    cudaMalloc(&d_mirror2s, mirrors_size);
    cudaMemcpy(d_mirrors, h_mirrors, mirrors_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mirror2s, h_mirror2s, mirrors_size, cudaMemcpyHostToDevice);

    int* d_adjacency_matrix = nullptr;
    size_t adjacency_size = num_nodes * max_degree * sizeof(int);

    if (h_adjacency_matrix != nullptr && max_degree > 0) {
        cudaMalloc(&d_adjacency_matrix, adjacency_size);
        cudaMemcpy(d_adjacency_matrix, h_adjacency_matrix, adjacency_size, cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < iterations; i++){
        printf(".");
        if (num_nodes < 5000) {
            // Use naive algorithm for repulsion for small graphs
            compute_repulsion_kernel_naive<<<gridSize, blockSize>>>(d_positions, d_velocities, num_nodes, repel);
            cudaDeviceSynchronize();
        } else {
            int num_bins = GRID_SIZE * GRID_SIZE * GRID_SIZE;

            // Host data for bounds and bin size
            glm::vec4 h_min_bounds(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
            glm::vec4 h_max_bounds(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

            Bin* d_bins;
            int* d_node_bins;

            size_t bin_size_bytes = num_bins * sizeof(Bin);
            size_t node_bins_size = num_nodes * sizeof(int);

            cudaMalloc(&d_bins, bin_size_bytes);
            cudaMalloc(&d_node_bins, node_bins_size);

            cudaMemset(d_bins, 0, bin_size_bytes);

            // Step 1: Compute AABB
            glm::vec4 *d_min_bounds, *d_max_bounds;
            cudaMalloc(&d_min_bounds, sizeof(glm::vec4));
            cudaMalloc(&d_max_bounds, sizeof(glm::vec4));
            cudaMemcpy(d_min_bounds, &h_min_bounds, sizeof(glm::vec4), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max_bounds, &h_max_bounds, sizeof(glm::vec4), cudaMemcpyHostToDevice);

            compute_aabb<<<gridSize, blockSize>>>(d_positions, num_nodes, d_min_bounds, d_max_bounds);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_min_bounds, d_min_bounds, sizeof(glm::vec4), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max_bounds, d_max_bounds, sizeof(glm::vec4), cudaMemcpyDeviceToHost);

            cudaFree(d_min_bounds);
            cudaFree(d_max_bounds);

            // Calculate bin size dynamically
            glm::vec4 h_bin_size = (h_max_bounds - h_min_bounds) / float(GRID_SIZE);

            // Step 2: Compute node bin mapping on the host
            int* h_node_bins = new int[num_nodes];
            compute_node_bins(h_positions, h_node_bins, num_nodes, h_min_bounds, h_bin_size);

            // Allocate memory for sorted positions and node bins
            glm::vec4* h_sorted_positions = new glm::vec4[num_nodes];
            int* h_sorted_indices = new int[num_nodes];

            // Populate bin counts and sort positions
            int bin_counts[num_bins] = {0};
            for (int i = 0; i < num_nodes; ++i) {
                bin_counts[h_node_bins[i]]++;
            }

            sort_positions_by_bins_with_indices(h_positions, h_sorted_positions, h_node_bins, 
                                                h_sorted_indices, num_nodes, bin_counts);

            // Copy sorted data to device
            glm::vec4 *d_sorted_positions;
            cudaMalloc(&d_sorted_positions, vec4_size);
            cudaMemcpy(d_sorted_positions, h_sorted_positions, vec4_size, cudaMemcpyHostToDevice);

            // Compute bin start indices using prefix sum on host
            int* h_bin_start_indices = new int[num_bins];
            h_bin_start_indices[0] = 0;
            for (int i = 1; i < num_bins; ++i) {
                h_bin_start_indices[i] = h_bin_start_indices[i - 1] + bin_counts[i - 1];
            }

            int* d_bin_start_indices;
            cudaMalloc(&d_bin_start_indices, num_bins * sizeof(int));
            cudaMemcpy(d_bin_start_indices, h_bin_start_indices, num_bins * sizeof(int), cudaMemcpyHostToDevice);

            int* d_sorted_indices;
            cudaMalloc(&d_sorted_indices, num_nodes * sizeof(int));
            cudaMemcpy(d_sorted_indices, h_sorted_indices, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

            // Step 3: Populate bins
            populate_bins<<<gridSize, blockSize>>>(d_sorted_positions, d_bins, num_nodes, h_min_bounds, h_bin_size);
            cudaDeviceSynchronize();

            // Step 4: Finalize bins
            finalize_bins<<<(num_bins + blockSize - 1) / blockSize, blockSize>>>(d_bins, num_bins);
            cudaDeviceSynchronize();

            // Step 5: Compute repulsion forces
            compute_repulsion_kernel_binned<<<gridSize, blockSize>>>(d_sorted_positions, d_velocities, d_bins, 
                                                                     d_bin_start_indices, d_sorted_indices, num_nodes, 
                                                                     h_min_bounds, h_bin_size, repel);
            cudaDeviceSynchronize();

            // Cleanup
            delete[] h_sorted_positions;
            delete[] h_sorted_indices;
            delete[] h_node_bins;
            delete[] h_bin_start_indices;
            cudaFree(d_bins);
            cudaFree(d_node_bins);
            cudaFree(d_sorted_indices);
            cudaFree(d_bin_start_indices);
        }

        attraction_kernel<<<gridSize, blockSize>>>(d_positions, d_adjacency_matrix, d_velocities, num_nodes, max_degree, attract);
        cudaDeviceSynchronize();

        mirror_kernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_mirrors, d_mirror2s, num_nodes, mirror_force, decay, dimension);
        cudaDeviceSynchronize();
    }
    cudaFree(d_mirror2s);
    cudaFree(d_mirrors);
    cudaFree(d_adjacency_matrix);

    // Copy final velocity deltas back to host
    cudaMemcpy(h_velocities, d_velocities, vec4_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_positions, d_positions, vec4_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_velocities);
    cudaFree(d_positions);
}
