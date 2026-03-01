// Given a pixel buffer and depth buffer, highlight edges based on depth discontinuities.
#include <cuda_runtime.h>
#include "color.cuh"

// Kernel over all pixels.
// If the depth difference between a pixel and any of its 4-connected neighbors
// is greater than .1, color the pixel with edge_color.
__global__ void cuda_edge_detect_kernel(
    uint32_t* d_pixels, float* d_depth_buffer, int w, int h, uint32_t edge_color
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float depth = d_depth_buffer[idx];
    if (depth == 0) return; // Ignore background

    // Check 4-connected neighbors
    float max_depth_diff = 0.0f;

    if (x > 0) {
        float neighbor_depth = d_depth_buffer[y * w + (x - 1)];
        max_depth_diff = fmax(max_depth_diff, fabs(depth - neighbor_depth));
    }
    if (x < w - 1) {
        float neighbor_depth = d_depth_buffer[y * w + (x + 1)];
        max_depth_diff = fmax(max_depth_diff, fabs(depth - neighbor_depth));
    }
    if (y > 0) {
        float neighbor_depth = d_depth_buffer[(y - 1) * w + x];
        max_depth_diff = fmax(max_depth_diff, fabs(depth - neighbor_depth));
    }
    if (y < h - 1) {
        float neighbor_depth = d_depth_buffer[(y + 1) * w + x];
        max_depth_diff = fmax(max_depth_diff, fabs(depth - neighbor_depth));
    }

    d_pixels[idx] = d_color_combine(d_pixels[idx], edge_color, fmin(max_depth_diff * 6.0f, 1.0f));
}

void cuda_edge_detect(
    uint32_t* d_pixels, float* d_depth_buffer, int w, int h, uint32_t edge_color
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    cuda_edge_detect_kernel<<<gridSize, blockSize>>>(
        d_pixels, d_depth_buffer, w, h, edge_color
    );
    cudaDeviceSynchronize();
}
