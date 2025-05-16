#include <cuda_runtime.h>
#include "../misc/cuda_color.cu" // Contains implementation of overlay_pixel

__global__ void overlay_kernel(
    unsigned int* background, const int bw, const int bh,
    unsigned int* foreground, const int fw, const int fh,
    const int dx, const int dy, const float opacity)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= fw * fh) return;

    int x = idx % fw;
    int y = idx / fw;

    overlay_pixel(x+dx, y+dy, foreground[y * fw + x], opacity, background, bw, bh);
}

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity)
{
    if (opacity == 0.0f) return;

    unsigned int* d_background = nullptr;
    unsigned int* d_foreground = nullptr;

    size_t bg_size = bw * bh * sizeof(unsigned int);
    size_t fg_size = fw * fh * sizeof(unsigned int);

    cudaMalloc((void**)&d_background, bg_size);
    cudaMemcpy(d_background, h_background, bg_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_foreground, fg_size);
    cudaMemcpy(d_foreground, h_foreground, fg_size, cudaMemcpyHostToDevice);

    int numPixels = fw * fh;
    int blockSize = 256;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;
    overlay_kernel<<<numBlocks, blockSize>>>(
        d_background, bw, bh,
        d_foreground, fw, fh,
        dx, dy, opacity);
    cudaDeviceSynchronize();

    cudaMemcpy(h_background, d_background, bg_size, cudaMemcpyDeviceToHost);

    cudaFree(d_background);
    cudaFree(d_foreground);
}
