// Render a grid of all n-state turing machines, plotting for any pixel
// the amount of steps it takes for the machine to halt

// For a m-symbol, n-state Turing machine, the amount of possible machines is
// given by (2m(n+1))^(mn)
// The following table shows the number of machines for small n:
// n | # machines
// 1 | 64
// 2 | 20,736
// 3 | 16,777,216
// 4 | 25,600,000,000
// 5 | 63,403,380,965,376

#include <cuda_runtime.h>
#include <stdio.h>
#include "../Host_Device_Shared/vec.h"
#include "common_graphics.cuh"

struct TuringMachine {
    bool left_right[10];
    int write_symbol[10];
    int next_state[10];
};

__device__ void decode_turing_machine_index(int x, int y, int grid_w, int grid_h, int num_states, int num_symbols, TuringMachine* tm) {
    int remaining_x = x;
    int remaining_y = y;
    int exponent = num_symbols * num_states;
    int w_base = num_states + 1;
    int h_base = 2 * num_symbols;
    for (int i = 0; i < exponent; i++) {
        grid_w /= w_base;
        grid_h /= h_base;
        int x_here = remaining_x / grid_w;
        int y_here = remaining_y / grid_h;
        remaining_x %= grid_w;
        remaining_y %= grid_h;
        tm->left_right[i] = (y_here % 2) == 1;
        tm->write_symbol[i] = y_here / 2;
        tm->next_state[i] = x_here - 1;
    }
}

__global__ void beaver_grid_kernel(int num_states, int num_symbols, unsigned int* pixels, int w, int h, int grid_w, int grid_h, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, int max_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int pixel_index = idy * w + idx;
    if(idx >= w || idy >= h) {
        return;
    }

    Cuda::vec2 point_vec = Cuda::pixel_to_point(Cuda::vec2(idx, idy), lx_ty, rx_by, Cuda::vec2(w, h));
    point_vec += Cuda::vec2(grid_w*.5, grid_h*.5); // Centering

    if(point_vec.x < 0 || point_vec.x >= grid_w || point_vec.y < 0 || point_vec.y >= grid_h) {
        return;
    }

    TuringMachine tm;
    decode_turing_machine_index(point_vec.x, point_vec.y, grid_w, grid_h, num_states, num_symbols, &tm);
    // Simulate the Turing machine and determine the number of steps until it halts
    int tape[1001] = {0};
    int head_position = 500;
    int current_state = 0;
    int steps = 0;
    while (steps < max_steps) {
        int action_index = current_state * num_symbols + tape[head_position];

        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        current_state = tm.next_state[action_index];

        steps++;
        if(current_state == -1) {
            break;
        }
    }

    bool halted = current_state == -1;

    //int col = 0xff000000 | (0x3f << ((head_position - 498) * 4));
    //int col = 0xff000000 | (0x3f << (current_state * 8));
    //pixels[pixel_index] = col;
    pixels[pixel_index] = halted ? 0xff000000 : 0xff00ff00;
}

extern "C" void beaver_grid_cuda(int num_states, int num_symbols, unsigned int* pixels, int w, int h, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, int max_steps) {
    unsigned int* d_pixels;
    int w_base = (num_states + 1);
    int h_base = 2 * num_symbols;
// given by (2m(n+1))^(mn)
    int exp = num_symbols * num_states;
    int grid_w = 1;
    int grid_h = 1;
    for (int i = 0; i < exp; i++) {
        grid_w *= w_base;
        grid_h *= h_base;
    }
    size_t size = w * h * sizeof(unsigned int);
    cudaMalloc(&d_pixels, size);
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    beaver_grid_kernel<<<gridSize, blockSize>>>(num_states, num_symbols, d_pixels, w, h, grid_w, grid_h, lx_ty, rx_by, max_steps);
    cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
