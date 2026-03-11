#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "color.cuh"

__global__
void alpha_overlay_kernel(unsigned int* pixels,
                          const Cuda::vec2 size,
                          int bg)
{
    // 2D coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size.x || y >= size.y) return;

    int idx = y * size.x + x;

    unsigned int pix = pixels[idx];

    pixels[idx] = d_color_combine(bg, pix);
}

extern "C"
void alpha_overlay_cuda(unsigned int* src_host,
                        const Cuda::vec2& size,
                        unsigned int bg_color)
{
    if (!src_host || size.x <= 0 || size.y <= 0) return;

    size_t npixels = (size_t)size.x * (size_t)size.y;
    size_t src_bytes = npixels * sizeof(uint32_t);

    uint32_t* d_src = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_src, src_bytes);
    cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice);

    // Launch configuration: 16x16 threads per block is common for 2D workloads
    dim3 threads(16, 16);
    dim3 blocks( (size.x + threads.x - 1) / threads.x,
                 (size.y + threads.y - 1) / threads.y );

    // Launch kernel
    alpha_overlay_kernel<<<blocks, threads>>>(d_src, size, bg_color);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(src_host, d_src, src_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
}

