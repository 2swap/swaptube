// Draws simple geometric shapes
#include <cuda_runtime.h>
#include "color.cuh"

__global__ void grid_kernel(uint32_t* pix, const Cuda::ivec2 wh, const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by, const uint32_t* grid, const Cuda::ivec2 grid_wh, const Cuda::vec2 grid_start)
{
    Cuda::ivec2 pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= wh.x || pos.y >= wh.y) return;

    Cuda::vec2 point_vec = pixel_to_point_in_screen(pos, lx_ty, rx_by, wh);
    Cuda::vec2 grid_pos = point_vec - grid_start;
    Cuda::ivec2 cell_pos = Cuda::ivec2(int(grid_pos.x), int(grid_pos.y));
    Cuda::vec2 grid_pos_decimal = grid_pos - cell_pos;

    if (grid_pos.x < 0 || grid_pos.y < 0 || grid_pos.x >= grid_wh.x || grid_pos.y >= grid_wh.y ||
        grid_pos_decimal.x < 0.05f || grid_pos_decimal.x > 0.95f || grid_pos_decimal.y < 0.05f || grid_pos_decimal.y > 0.95f) {
        pix[pos.y * wh.x + pos.x] = 0x00000000;
        return;
    }

    uint32_t cell_value = grid[int(grid_pos.y) * grid_wh.x + int(grid_pos.x)];
    pix[pos.y * wh.x + pos.x] = cell_value;
}

extern "C" void draw_grid(uint32_t* pix, const Cuda::ivec2& wh, const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by, const uint32_t* grid, const Cuda::ivec2& grid_wh, const Cuda::vec2& grid_start) {
    // Copy grid data to device memory
    uint32_t* d_grid;
    size_t grid_size = grid_wh.x * grid_wh.y * sizeof(uint32_t);
    cudaMalloc(&d_grid, grid_size);
    cudaMemcpy(d_grid, grid, grid_size, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    grid_kernel<<<gridSize, blockSize>>>(pix, wh, lx_ty, rx_by, d_grid, grid_wh, grid_start);

    // Clean up device memory
    cudaFree(d_grid);
}
