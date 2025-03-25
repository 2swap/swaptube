#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define GRID_SIZE 10 // 10x10x10 bins
#define BIN_INDEX(x, y, z) ((z) * GRID_SIZE * GRID_SIZE + (y) * GRID_SIZE + (x))

struct Bin {
    int count;
    glm::dvec4 center_of_mass;
};

__device__ glm::dvec4 compute_force(glm::dvec4 pos_i, glm::dvec4 pos_j) {
    glm::dvec4 diff = pos_i - pos_j;
    double dist_sq = glm::dot(diff, diff) + 1;
    return glm::normalize(diff) / (dist_sq * 10 + 2);
}

__global__ void compute_repulsion_kernel_naive(const glm::dvec4* positions, glm::dvec4* velocity_deltas,
                                               int num_nodes, double repel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    glm::dvec4 pos_i = positions[i];
    glm::dvec4 delta = glm::dvec4(0.0);

    for (int j = 0; j < num_nodes; ++j) {
        if (i == j) continue;

        delta += repel * compute_force(pos_i, positions[j]);
    }

    velocity_deltas[i] = delta;
}


void sort_positions_by_bins_with_indices(const glm::dvec4* positions, glm::dvec4* sorted_positions, 
                                         const int* node_bins, int* sorted_node_bins, 
                                         int* original_indices, int* sorted_indices,
                                         int num_nodes, int* bin_counts) {
    // Step 1: Compute cumulative bin counts (prefix sum) to determine sorted indices
    int num_bins = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    int* bin_offsets = new int[num_bins + 1];
    bin_offsets[0] = 0;

    for (int i = 0; i < num_bins; ++i) {
        bin_offsets[i + 1] = bin_offsets[i] + bin_counts[i];
    }

    // Step 2: Insert nodes into their sorted positions
    for (int i = 0; i < num_nodes; ++i) {
        int bin_idx = node_bins[i];
        int sorted_idx = bin_offsets[bin_idx]++; // Get the current position and increment the offset

        sorted_positions[sorted_idx] = positions[i];
        sorted_node_bins[sorted_idx] = bin_idx;
        sorted_indices[sorted_idx] = original_indices[i];
    }

    // Cleanup
    delete[] bin_offsets;
}

__global__ void compute_repulsion_kernel_binned(const glm::dvec4* sorted_positions, glm::dvec4* velocity_deltas,
                                                const Bin* bins, const int* bin_start_indices, 
                                                const int* sorted_indices, int num_nodes, 
                                                glm::dvec4 min_bounds, glm::dvec4 bin_size, double repel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    glm::dvec4 pos_i = sorted_positions[i];
    glm::dvec4 delta = glm::dvec4(0.0);

    // Determine the bin index for the current node
    glm::ivec3 bin_idx = glm::ivec3((pos_i - min_bounds) / bin_size);
    bin_idx = glm::clamp(bin_idx, glm::ivec3(0), glm::ivec3(GRID_SIZE - 1));

    // **Step 1: Interact with nodes in neighboring bins**
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

                int neighbor_bin_flat_idx = BIN_INDEX(neighbor_bin_idx.x, neighbor_bin_idx.y, neighbor_bin_idx.z);
                int start_idx = bin_start_indices[neighbor_bin_flat_idx];
                int end_idx = bin_start_indices[neighbor_bin_flat_idx + 1];

                for (int j = start_idx; j < end_idx; ++j) {
                    if (i == j) continue; // Skip self-interaction

                    delta += repel * compute_force(pos_i, sorted_positions[j]);
                }
            }
        }
    }

    // **Step 2: Interact with non-neighboring bins (bin-level interaction)**
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

                int non_neighbor_bin_idx = BIN_INDEX(current_bin_idx_3d.x, current_bin_idx_3d.y, current_bin_idx_3d.z);
                Bin bin = bins[non_neighbor_bin_idx];

                if (bin.count == 0) continue; // Skip empty bins

                delta += (double)bin.count * repel * compute_force(pos_i, bin.center_of_mass);
            }
        }
    }

    // Map back to the original index
    int original_idx = sorted_indices[i];
    velocity_deltas[original_idx] = delta;
}

__device__ double atomicMin_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) break; // If the current value is already smaller, no need to update
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicMax_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break; // If the current value is already larger, no need to update
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void populate_bins(const glm::dvec4* positions, Bin* bins, int num_nodes, glm::dvec4 min_bounds, glm::dvec4 bin_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    glm::dvec4 pos = positions[i];
    glm::ivec3 bin_idx = glm::ivec3((pos - min_bounds) / bin_size);
    bin_idx = glm::clamp(bin_idx, glm::ivec3(0), glm::ivec3(GRID_SIZE - 1));

    int bin_flat_idx = BIN_INDEX(bin_idx.x, bin_idx.y, bin_idx.z);

    // Atomic operations to update the bin's data
    atomicAdd(&bins[bin_flat_idx].count, 1);
    atomicAdd_double(&bins[bin_flat_idx].center_of_mass.x, pos.x);
    atomicAdd_double(&bins[bin_flat_idx].center_of_mass.y, pos.y);
    atomicAdd_double(&bins[bin_flat_idx].center_of_mass.z, pos.z);
    atomicAdd_double(&bins[bin_flat_idx].center_of_mass.w, pos.w);
}

__global__ void finalize_bins(Bin* bins, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bins) return;

    if (bins[i].count > 0) {
        bins[i].center_of_mass /= double(bins[i].count);
    }
}

struct AABB {
    glm::dvec4 min_bounds;
    glm::dvec4 max_bounds;
};

__global__ void compute_aabb(const glm::dvec4* positions, int num_nodes, glm::dvec4* min_bounds, glm::dvec4* max_bounds) {
    __shared__ glm::dvec4 local_min;
    __shared__ glm::dvec4 local_max;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid == 0) {
        local_min = glm::dvec4(DBL_MAX);
        local_max = glm::dvec4(-DBL_MAX);
    }
    __syncthreads();

    // Update local min/max
    if (i < num_nodes) {
        atomicMin_double(&local_min.x, positions[i].x);
        atomicMin_double(&local_min.y, positions[i].y);
        atomicMin_double(&local_min.z, positions[i].z);

        atomicMax_double(&local_max.x, positions[i].x);
        atomicMax_double(&local_max.y, positions[i].y);
        atomicMax_double(&local_max.z, positions[i].z);
    }
    __syncthreads();

    // Update global min/max using shared memory
    if (tid == 0) {
        atomicMin_double(&min_bounds->x, local_min.x);
        atomicMin_double(&min_bounds->y, local_min.y);
        atomicMin_double(&min_bounds->z, local_min.z);

        atomicMax_double(&max_bounds->x, local_max.x);
        atomicMax_double(&max_bounds->y, local_max.y);
        atomicMax_double(&max_bounds->z, local_max.z);
    }
}

void compute_node_bins(const glm::dvec4* positions, int* node_bins, int num_nodes, glm::dvec4 min_bounds, glm::dvec4 bin_size) {
    for (int i = 0; i < num_nodes; ++i) {
        glm::ivec3 bin_idx = glm::ivec3((positions[i] - min_bounds) / bin_size);
        bin_idx = glm::clamp(bin_idx, glm::ivec3(0), glm::ivec3(GRID_SIZE - 1));
        node_bins[i] = BIN_INDEX(bin_idx.x, bin_idx.y, bin_idx.z);
    }
}

extern "C" void compute_repulsion_cuda(const glm::dvec4* host_positions, glm::dvec4* host_velocity_deltas, 
                                       int num_nodes, double repel) {
    glm::dvec4 *d_positions, *d_velocity_deltas;

    size_t size = num_nodes * sizeof(glm::dvec4);

    // Allocate device memory
    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_velocity_deltas, size);

    // Copy positions to device
    cudaMemcpy(d_positions, host_positions, size, cudaMemcpyHostToDevice);
    cudaMemset(d_velocity_deltas, 0, size);

    int blockSize = 128;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;

    if (num_nodes < 5000) {
        // Use naive algorithm for small graphs
        compute_repulsion_kernel_naive<<<gridSize, blockSize>>>(d_positions, d_velocity_deltas, num_nodes, repel);
    } else {
        // Use binned algorithm for larger graphs

        // Host data for bounds and bin size
        glm::dvec4 h_min_bounds(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
        glm::dvec4 h_max_bounds(-DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX);
        glm::dvec4 h_bin_size;

        Bin* d_bins;
        int* d_node_bins;

        size_t bin_size = GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(Bin);
        size_t node_bins_size = num_nodes * sizeof(int);

        cudaMalloc(&d_bins, bin_size);
        cudaMalloc(&d_node_bins, node_bins_size);

        cudaMemset(d_bins, 0, bin_size);

        // Step 1: Compute AABB
        glm::dvec4 *d_min_bounds, *d_max_bounds;
        cudaMalloc(&d_min_bounds, sizeof(glm::dvec4));
        cudaMalloc(&d_max_bounds, sizeof(glm::dvec4));
        cudaMemcpy(d_min_bounds, &h_min_bounds, sizeof(glm::dvec4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_bounds, &h_max_bounds, sizeof(glm::dvec4), cudaMemcpyHostToDevice);

        compute_aabb<<<gridSize, blockSize>>>(d_positions, num_nodes, d_min_bounds, d_max_bounds);
        cudaMemcpy(&h_min_bounds, d_min_bounds, sizeof(glm::dvec4), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max_bounds, d_max_bounds, sizeof(glm::dvec4), cudaMemcpyDeviceToHost);

        cudaFree(d_min_bounds);
        cudaFree(d_max_bounds);

        // Calculate bin size dynamically
        h_bin_size = (h_max_bounds - h_min_bounds) / double(GRID_SIZE);

        // Step 2: Compute node bin mapping on the host
        int* host_node_bins = new int[num_nodes];
        compute_node_bins(host_positions, host_node_bins, num_nodes, h_min_bounds, h_bin_size);

        // Allocate memory for sorted positions and node bins
        glm::dvec4* sorted_positions = new glm::dvec4[num_nodes];
        int* sorted_node_bins = new int[num_nodes];
        int* original_indices = new int[num_nodes];
        int* sorted_indices = new int[num_nodes];

        for (int i = 0; i < num_nodes; ++i) {
            original_indices[i] = i;
        }

        // Populate bin counts and sort positions
        int bin_counts[GRID_SIZE * GRID_SIZE * GRID_SIZE] = {0};
        for (int i = 0; i < num_nodes; ++i) {
            bin_counts[host_node_bins[i]]++;
        }

        sort_positions_by_bins_with_indices(host_positions, sorted_positions, host_node_bins, 
                                            sorted_node_bins, original_indices, sorted_indices, 
                                            num_nodes, bin_counts);

        // Copy sorted data to device
        cudaMemcpy(d_positions, sorted_positions, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_bins, sorted_node_bins, node_bins_size, cudaMemcpyHostToDevice);

        int* d_sorted_indices;
        cudaMalloc(&d_sorted_indices, num_nodes * sizeof(int));
        cudaMemcpy(d_sorted_indices, sorted_indices, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        // Step 3: Populate bins
        populate_bins<<<gridSize, blockSize>>>(d_positions, d_bins, num_nodes, h_min_bounds, h_bin_size);

        // Step 4: Finalize bins
        finalize_bins<<<(GRID_SIZE * GRID_SIZE * GRID_SIZE + blockSize - 1) / blockSize, blockSize>>>(d_bins, GRID_SIZE * GRID_SIZE * GRID_SIZE);

        // Step 5: Compute repulsion forces
        compute_repulsion_kernel_binned<<<gridSize, blockSize>>>(d_positions, d_velocity_deltas, d_bins, 
                                                                 d_node_bins, d_sorted_indices, num_nodes, 
                                                                 h_min_bounds, h_bin_size, repel);

        // Cleanup
        delete[] sorted_positions;
        delete[] sorted_node_bins;
        delete[] original_indices;
        delete[] host_node_bins;
        cudaFree(d_bins);
        cudaFree(d_node_bins);
        cudaFree(d_sorted_indices);
    }

    // Copy results back to host
    cudaMemcpy(host_velocity_deltas, d_velocity_deltas, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_velocity_deltas);
}
