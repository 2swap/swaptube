#include <cuda_runtime.h>

__global__ void convolve_map_kernel(const unsigned int* a, const int aw, const int ah, const unsigned int* b, const int bw, const int bh, unsigned int* map, const int mapw, const int maph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mapw || y >= maph) return;

    unsigned int sum = 0;

    int shift_x = -bw + 1 + x;
    int shift_y = -bh + 1 + y;
    int x_min = max(0 , shift_x   );
    int x_max = min(aw, shift_x+bw);
    int y_min = max(0 , shift_y   );
    int y_max = min(ah, shift_y+bh);
    for (int dx = x_min; dx < x_max; ++dx) {
        for (int dy = y_min; dy < y_max; ++dy) {
            int a_alpha = a[ dy          * aw +  dx         ] >> 24;
            int b_alpha = b[(dy+shift_y) * bw + (dx+shift_x)] >> 24;

            sum+= (a_alpha * b_alpha) >> 8;
        }
    }

    map[x + y * mapw] = sum;
}

extern "C" void convolve_map_cuda(const unsigned int* a, const int aw, const int ah, const unsigned int* b, const int bw, const int bh, unsigned int* map, const int mapw, const int maph) {
    unsigned int* d_a, * d_b, * d_map;

    size_t size_a = aw * ah * sizeof(unsigned int);
    size_t size_b = bw * bh * sizeof(unsigned int);
    size_t size_map = mapw * maph * sizeof(unsigned int);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_map, size_map);

    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    cudaMemset(d_map, -1, size_map);

    dim3 blockSize(16, 16);
    dim3 gridSize((mapw + blockSize.x - 1) / blockSize.x, (maph + blockSize.y - 1) / blockSize.y);

    convolve_map_kernel<<<gridSize, blockSize>>>(d_a, aw, ah, d_b, bw, bh, d_map, mapw, maph);

    cudaMemcpy(map, d_map, size_map, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_map);
}
