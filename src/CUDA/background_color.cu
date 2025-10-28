#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "color.cuh"

__global__
void alpha_overlay_kernel(unsigned int* pixels,
                          int width, int height,
                          int bg)
{
    // 2D coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    unsigned int pix = pixels[idx];

    pixels[idx] = d_color_combine(bg, pix);
}

extern "C"
void alpha_overlay_cuda(unsigned int* src_host,
                        int width, int height,
                        unsigned int bg_color)
{
    if (!src_host || width <= 0 || height <= 0) return;

    size_t npixels = (size_t)width * (size_t)height;
    size_t src_bytes = npixels * sizeof(uint32_t);

    uint32_t* d_src = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_src, src_bytes);
    cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice);

    // Launch configuration: 16x16 threads per block is common for 2D workloads
    dim3 threads(16, 16);
    dim3 blocks( (width  + threads.x - 1) / threads.x,
                 (height + threads.y - 1) / threads.y );

    // Launch kernel
    alpha_overlay_kernel<<<blocks, threads>>>(d_src, width, height, bg_color);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(src_host, d_src, src_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
}

