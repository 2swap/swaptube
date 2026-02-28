#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/vec.h"

#define GRID_SIZE 10 // 10x10x10 bins
#define BIN_INDEX(a) ((a.z) * GRID_SIZE * GRID_SIZE + (a.y) * GRID_SIZE + (a.x))

struct Bin {
    int count;
    Cuda::vec4 center_of_mass;
};

__device__ Cuda::vec4 compute_force(Cuda::vec4 pos_i, Cuda::vec4 pos_j) {
    Cuda::vec4 diff = pos_i - pos_j;
    float dist_sq = dot(diff, diff) + 1.0f;
    Cuda::vec4 norm = normalize(diff);
    Cuda::vec4 result = norm / (dist_sq * 10.0f + 2.0f);
    return result;
}

__global__ void compute_repulsion_kernel_naive(const Cuda::vec4* positions, Cuda::vec4* velocities, int num_nodes, float repel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    Cuda::vec4 pos_i = positions[i];
    Cuda::vec4 delta(0.0f);

    for (int j = 0; j < num_nodes; ++j) {
        if (i == j) continue;
        delta += compute_force(pos_i, positions[j]);
    }

    velocities[i] += delta * repel;
}


void sort_positions_by_bins_with_indices(const Cuda::vec4* positions, Cuda::vec4* sorted_positions, 
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

__global__ void compute_repulsion_kernel_binned(const Cuda::vec4* sorted_positions, Cuda::vec4* velocities,
                                                const Bin* bins, const int* bin_start_indices, 
                                                const int* sorted_indices, int num_nodes, 
                                                Cuda::vec4 min_bounds, Cuda::vec4 bin_size, float repel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    Cuda::vec4 pos_i = sorted_positions[i];
    Cuda::vec4 delta = Cuda::vec4(0.0f);

    // Determine the bin index for the current node
    Cuda::vec3 bin_idx = integerize((pos_i - min_bounds) / bin_size);
    bin_idx = Cuda::clamp(bin_idx, Cuda::vec3(0), Cuda::vec3(GRID_SIZE - 1));

    // Step 1: Interact with nodes in neighboring bins
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                Cuda::vec3 neighbor_bin_idx = bin_idx + Cuda::vec3(dx, dy, dz);

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
                Cuda::vec3 current_bin_idx_3d(x, y, z);

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

__global__ void populate_bins(const Cuda::vec4* positions, Bin* bins, int num_nodes, Cuda::vec4 min_bounds, Cuda::vec4 bin_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    Cuda::vec4 pos = positions[i];
    Cuda::vec3 bin_idx = integerize((pos - min_bounds) / bin_size);
    bin_idx = Cuda::clamp(bin_idx, Cuda::vec3(0), Cuda::vec3(GRID_SIZE - 1));

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

__global__ void compute_aabb(const Cuda::vec4* positions, int num_nodes, Cuda::vec4* min_bounds, Cuda::vec4* max_bounds) {
    __shared__ Cuda::vec4 local_min;
    __shared__ Cuda::vec4 local_max;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        local_min = Cuda::vec4(FLT_MAX);
        local_max = Cuda::vec4(-FLT_MAX);
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

void compute_node_bins(const Cuda::vec4* positions, int* node_bins, int num_nodes, Cuda::vec4 min_bounds, Cuda::vec4 bin_size) {
    for (int i = 0; i < num_nodes; ++i) {
        Cuda::vec3 bin_idx = integerize((positions[i] - min_bounds) / bin_size);
        bin_idx = Cuda::clamp(bin_idx, Cuda::vec3(0), Cuda::vec3(GRID_SIZE - 1));
        node_bins[i] = BIN_INDEX(bin_idx);
    }
}

// Compute attraction force helper function on device
// Return float attraction scalar
__device__ float get_attraction_force (float dist_sq) {
    float dist_6th = dist_sq * dist_sq * dist_sq * .05f;
    return (dist_6th - 1.0f) / (dist_6th + 1.0f) * .2f - .1f;
};

// Kernel to compute mirror forces
__global__ void mirror_kernel(Cuda::vec4* positions, Cuda::vec4* velocities, const int* adjacency_matrix, const int* mirrors, const int* mirror2s, int num_nodes, float mirror_force, const float decay, const float dimension, const float attract, const int max_degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    if (adjacency_matrix != nullptr && max_degree > 0) {
        Cuda::vec4 pos_i = positions[i];
        Cuda::vec4 delta(0.0f);

        for (int k = 0; k < max_degree; ++k) {
            int neighbor_idx = adjacency_matrix[i * max_degree + k];
            if (neighbor_idx < 0 || neighbor_idx >= num_nodes) {
                break;
            }
            Cuda::vec4 pos_j = positions[neighbor_idx];
            Cuda::vec4 diff = pos_i - pos_j;
            float dist_sq = dot(diff, diff) + 1.0f;

            delta += diff * get_attraction_force(dist_sq);
        }

        velocities[i] -= attract * delta;
    }

    if (mirror_force > 0.0000001) {
        Cuda::vec4 pos_i = positions[i];
        Cuda::vec4 delta(0.0f);

        {
            int index = mirrors[i];
            if(index >= 0 && index != i){
                Cuda::vec4 mirror = positions[index];
                mirror.x *= -1;
                delta += mirror - pos_i;
            }
        }

        {
            int index = mirror2s[i];
            if(index >= 0 && index != i){
                Cuda::vec4 mirror = positions[index];
                mirror.y *= -1;
                delta.x = 0;
                delta += mirror - pos_i;
            }
        }

        delta.w = 0;
        positions[i] += mirror_force * delta;
    }

    if(hasnan(velocities[i])) { velocities[i] = Cuda::vec4(0, 0, 0, 0); }
    if(hasnan( positions[i])) {  positions[i] = Cuda::vec4(0, 0, 0, 0); }

    float speedlimit = 20;
    float magnitude = length(velocities[i]);
    if (magnitude > speedlimit) {
        float scale = speedlimit / magnitude;
        velocities[i] *= scale;
    }

    velocities[i] *= decay;
    positions[i] += velocities[i];
    positions[i].z *= Cuda::clamp(dimension - 2, 0, 1);
    positions[i].w *= Cuda::clamp(dimension - 3, 0, 1);
}

extern "C" void compute_repulsion_cuda(Cuda::vec4* h_positions, Cuda::vec4* h_velocities, const int* h_adjacency_matrix, const int* h_mirrors, const int* h_mirror2s, int num_nodes, int max_degree, float attract, float repel, float mirror_force, const float decay, const float dimension, const int iterations) {
    if(num_nodes < 0) return;

    Cuda::vec4 *d_positions;
    Cuda::vec4 *d_velocities;
    size_t vec4_size = num_nodes * sizeof(Cuda::vec4);
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

    if (num_nodes < 5000) {
        for(int i = 0; i < iterations; i++){
            printf(".");
            fflush(stdout);
            compute_repulsion_kernel_naive<<<gridSize, blockSize>>>(d_positions, d_velocities, num_nodes, repel);
            cudaDeviceSynchronize();
            mirror_kernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_adjacency_matrix, d_mirrors, d_mirror2s, num_nodes, mirror_force, decay, dimension, attract, max_degree);
            cudaDeviceSynchronize();
        }
    } else {
        int num_bins = GRID_SIZE * GRID_SIZE * GRID_SIZE;

        // Host data for bounds and bin size
        Cuda::vec4 h_min_bounds(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
        Cuda::vec4 h_max_bounds(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

        Bin* d_bins;
        int* d_node_bins;

        size_t bin_size_bytes = num_bins * sizeof(Bin);
        size_t node_bins_size = num_nodes * sizeof(int);

        cudaMalloc(&d_bins, bin_size_bytes);
        cudaMalloc(&d_node_bins, node_bins_size);

        cudaMemset(d_bins, 0, bin_size_bytes);

        // Step 1: Compute AABB
        Cuda::vec4 *d_min_bounds, *d_max_bounds;
        cudaMalloc(&d_min_bounds, sizeof(Cuda::vec4));
        cudaMalloc(&d_max_bounds, sizeof(Cuda::vec4));
        cudaMemcpy(d_min_bounds, &h_min_bounds, sizeof(Cuda::vec4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_bounds, &h_max_bounds, sizeof(Cuda::vec4), cudaMemcpyHostToDevice);

        compute_aabb<<<gridSize, blockSize>>>(d_positions, num_nodes, d_min_bounds, d_max_bounds);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_min_bounds, d_min_bounds, sizeof(Cuda::vec4), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max_bounds, d_max_bounds, sizeof(Cuda::vec4), cudaMemcpyDeviceToHost);

        cudaFree(d_min_bounds);
        cudaFree(d_max_bounds);

        // Calculate bin size dynamically
        Cuda::vec4 h_bin_size = (h_max_bounds - h_min_bounds) / float(GRID_SIZE);

        // Step 2: Compute node bin mapping on the host
        int* h_node_bins = new int[num_nodes];
        compute_node_bins(h_positions, h_node_bins, num_nodes, h_min_bounds, h_bin_size);

        // Allocate memory for sorted positions and node bins
        Cuda::vec4* h_sorted_positions = new Cuda::vec4[num_nodes];
        int* h_sorted_indices = new int[num_nodes];

        // Populate bin counts and sort positions
        int bin_counts[num_bins] = {0};
        for (int i = 0; i < num_nodes; ++i) {
            bin_counts[h_node_bins[i]]++;
        }

        sort_positions_by_bins_with_indices(h_positions, h_sorted_positions, h_node_bins, 
                                            h_sorted_indices, num_nodes, bin_counts);

        // Copy sorted data to device
        Cuda::vec4 *d_sorted_positions;
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
        for(int i = 0; i < iterations; i++){
            compute_repulsion_kernel_binned<<<gridSize, blockSize>>>(d_sorted_positions, d_velocities, d_bins, 
                                                                     d_bin_start_indices, d_sorted_indices, num_nodes, 
                                                                     h_min_bounds, h_bin_size, repel);
            cudaDeviceSynchronize();
            mirror_kernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_adjacency_matrix, d_mirrors, d_mirror2s, num_nodes, mirror_force, decay, dimension, attract, max_degree);
            cudaDeviceSynchronize();
            printf(",");
            fflush(stdout);
        }

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
