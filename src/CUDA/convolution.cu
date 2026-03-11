#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"

__global__ void convolve_map_kernel(
    const unsigned int* a, const Cuda::vec2 a_size,
    const unsigned int* b, const Cuda::vec2 b_size,
    unsigned int* map, const Cuda::vec2 map_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= map_size.x || y >= map_size.y) return;

    unsigned int sum = 0;
    const Cuda::vec2 xy(x, y);

    const Cuda::vec2 shift = -b_size + xy + Cuda::vec2(1,1);
    const Cuda::vec2 minie = vec_max(0, shift);
    const Cuda::vec2 maxie = vec_min(a_size, shift+b_size);
    for (int dx = minie.x; dx < maxie.x; ++dx) {
        for (int dy = minie.y; dy < maxie.y; ++dy) {
            unsigned int a_alpha = a[ dy          * (int)a_size.x +  dx         ] >> 24;
            unsigned int b_alpha = b[(int)(dy-shift.y) * (int)b_size.x + (int)(dx-shift.x)] >> 24;

            //sum+= (a_alpha * b_alpha) >> 8;
            sum+= (a_alpha > 0 && b_alpha > 0) ? 1 : 0;
        }
    }

    map[x + y * (int)map_size.x] = sum;
}

extern "C" void convolve_map_cuda(
    const unsigned int* a, const Cuda::vec2& a_size,
    const unsigned int* b, const Cuda::vec2& b_size,
    unsigned int* map, const Cuda::vec2& map_size)
{
    unsigned int* d_a, * d_b, * d_map;

    size_t size_a = a_size.x * a_size.y * sizeof(unsigned int);
    size_t size_b = b_size.x * b_size.y * sizeof(unsigned int);
    size_t size_map = map_size.x * map_size.y * sizeof(unsigned int);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_map, size_map);

    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    cudaMemset(d_map, -1, size_map);

    dim3 blockSize(16, 16);
    dim3 gridSize((map_size.x + blockSize.x - 1) / blockSize.x, (map_size.y + blockSize.y - 1) / blockSize.y);

    convolve_map_kernel<<<gridSize, blockSize>>>(d_a, a_size, d_b, b_size, d_map, map_size);

    cudaMemcpy(map, d_map, size_map, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_map);
}
