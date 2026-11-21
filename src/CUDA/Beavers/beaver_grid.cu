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
#include <math.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../common_graphics.cuh"

// Transition slot i of the grid is state i % num_states reading symbol i / num_states.
// Along x each slot picks the next state (0 = halt), along y the write symbol and move.
__device__ void decode_turing_machine_index(int x, int y, int grid_w, int grid_h, int num_states, Cuda::TuringMachine* tm) {
    int remaining_x = x;
    int remaining_y = y;
    int exponent = 2 * num_states;
    int w_base = num_states + 1;
    int h_base = 4;
    for (int i = 0; i < exponent; i++) {
        grid_w /= w_base;
        grid_h /= h_base;
        int x_here = remaining_x / grid_w;
        int y_here = remaining_y / grid_h;
        remaining_x %= grid_w;
        remaining_y %= grid_h;
        const int next_state = x_here == 0 ? Cuda::TM_HALT : x_here - 1;
        tm->table[Cuda::tm_codon(i % num_states, i / num_states)] = Cuda::tm_pack(next_state, y_here / 2, y_here % 2);
    }
}

__global__ void beaver_grid_kernel(int num_states, uint32_t* pixels, const Cuda::ivec2 wh, int grid_w, int grid_h, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, int max_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int pixel_index = idy * wh.x + idx;
    if(idx >= wh.x || idy >= wh.y) {
        return;
    }

    Cuda::vec2 point_vec = Cuda::pixel_to_point_in_screen(Cuda::vec2(idx, idy), lx_ty, rx_by, wh);
    //point_vec *= Cuda::vec2(grid_w, grid_h);
    point_vec += Cuda::vec2(grid_w * .5, grid_h * .5); // Centering

    if(point_vec.x < 0 || point_vec.x >= grid_w || point_vec.y < 0 || point_vec.y >= grid_h) {
        return;
    }

    Cuda::TuringMachine tm;
    decode_turing_machine_index(point_vec.x, point_vec.y, grid_w, grid_h, num_states, &tm);
    // Simulate the Turing machine and determine the number of steps until it halts
    uint8_t tape[1001] = {0};
    int head_position = 500;
    int current_state = 0;
    int steps = 0;
    bool halted = false;
    while (steps < max_steps) {
        const int tr = tm.table[Cuda::tm_codon(current_state, tape[head_position])];

        // a halting transition still writes and moves
        tape[head_position] = Cuda::tm_write_symbol(tr);
        head_position += 2 * Cuda::tm_move_right(tr) - 1;
        if (Cuda::tm_next_state(tr) == Cuda::TM_HALT) {
            halted = true;
            break;
        }
        current_state = Cuda::tm_next_state(tr);
        steps++;
    }

    /* int num_ones = 0;
    for (int i = 0; i < 1001; i++) {
        if (tape[i] == 1) {
            num_ones++;
        }
    }*/

    //int col = 0xff000000 | (0x3f << ((head_position - 498) * 4));
    //int col = 0xff000000 | (0x3f << (current_state * 8));
    //pixels[pixel_index] = halted ? Cuda::rainbow((float) head_position / 42) : 0xff000000;
    float atan_steps = atanf(steps / 4.) / 1.57079632679f; // Normalize to [0, 1]
    pixels[pixel_index] = halted ? Cuda::rainbow(atan_steps) : 0xff000000;
}

extern "C" void beaver_grid_cuda(int num_states, uint32_t* pixels, const Cuda::ivec2& wh, const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by, int max_steps) {
    uint32_t* d_pixels;
    int w_base = (num_states + 1);
    int h_base = 4;
// given by (2m(n+1))^(mn)
    int exp = 2 * num_states;
    int grid_w = 1;
    int grid_h = 1;
    for (int i = 0; i < exp; i++) {
        grid_w *= w_base;
        grid_h *= h_base;
    }
    size_t size = wh.x * wh.y * sizeof(uint32_t);
    cudaMalloc(&d_pixels, size);
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);

    beaver_grid_kernel<<<gridSize, blockSize>>>(num_states, d_pixels, wh, grid_w, grid_h, lx_ty, rx_by, max_steps);
    cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
