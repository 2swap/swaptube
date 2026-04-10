#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/TuringMachine.h"
#include "common_graphics.cuh"

constexpr int half_tape_length = 20;

__device__ void get_child(TuringMachine& tm, int action_index, const Cuda::vec2& pixel, Cuda::vec2& lx_ty, Cuda::vec2& rx_by) {
    bool symbols_split_rows = true;
    bool dirs_split_rows = true;
    bool states_split_rows = false;
    int num_cols = (symbols_split_rows ? 1 : tm.num_symbols) * (dirs_split_rows ? 1 : 2) * (states_split_rows ? 1 : tm.num_states);
    int num_rows = (2 * tm.num_symbols * tm.num_states) / num_cols;
    Cuda::vec2 child_size = (rx_by - lx_ty) / Cuda::vec2(num_cols, num_rows);

    /* ivecs don't exist yet
    Cuda::ivec2 child_pos = floor((pixel - lx_ty) / child_size);
    lx_ty += child_size * child_pos;
    rx_by = lx_ty + child_size;
    tm.left_right[action_index] = (dirs_split_rows ? child_pos.y : child_pos.x) % 2;
    tm.write_symbol[action_index] = ((symbols_split_rows ? child_pos.y : child_pos.x) / (dirs_split_rows == symbols_split_rows ? 2 : 1)) % tm.num_symbols;
    tm.next_state[action_index] = states_split_rows ? (child_pos.y * num_states) / num_rows : (child_pos.x * num_states) / num_cols;
    tm.num_symbols += (int)(tm.write_symbol[action_index] == tm.num_symbols-1);
    tm.num_states += (int)(tm.next_state[action_index] == tm.num_states-1);
    */

    int child_x = floor((pixel.x - lx_ty.x) / child_size.x);
    int child_y = floor((pixel.y - lx_ty.y) / child_size.y);
    lx_ty += child_size * Cuda::vec2(child_x, child_y);
    rx_by = lx_ty + child_size;
    tm.left_right[action_index] = (dirs_split_rows ? child_y : child_x) % 2;
    tm.write_symbol[action_index] = ((symbols_split_rows ? child_y : child_x) / (dirs_split_rows == symbols_split_rows ? 2 : 1)) % tm.num_symbols;
    tm.next_state[action_index] = states_split_rows ? (child_y * tm.num_states) / num_rows : (child_x * tm.num_states) / num_cols;
    tm.num_symbols += (int)(tm.write_symbol[action_index] == tm.num_symbols-1);
    tm.num_states += (int)(tm.next_state[action_index] == tm.num_states-1);
}

__global__ void beaver_TNF_kernel(Color* pixels, int w, int h, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, bool corners_in_01, float border_thickness, TuringMachine TNF_root, int max_steps) {
    // convert the corners so that instead of storing the screen corners in the CoordinateScene's coordinate system, they store the TNF grid corners in the screen's coordinate system
    Cuda::vec2 grid_lx_ty = -lx_ty / (rx_by - lx_ty);
    rx_by = (Cuda::vec2(1,1) - lx_ty) / (rx_by - lx_ty);
    lx_ty = grid_lx_ty;
    // convert from [0,1] to pixels
    lx_ty *= Cuda::vec2(w, h);
    rx_by *= Cuda::vec2(w, h);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= w || idy >= h) {
        return;
    }
    // flip the resulting image vertically to match CoordinateScene's coordinates
    int pixel_index = (h - 1 - idy) * w + idx;

    if (!(lx_ty.x < idx && idx < rx_by.x && lx_ty.y < idy && idy < rx_by.y)) {
        pixels[pixel_index] = 0;
        return;
    }

    TuringMachine tm;
    tm.num_symbols = TNF_root.num_symbols;
    tm.num_states = TNF_root.num_states;
    for (int i=0; i<CODON_MEM_LIMIT; i++) {
        tm.left_right[i] = TNF_root.left_right[i];
        tm.write_symbol[i] = TNF_root.write_symbol[i];
        tm.next_state[i] = TNF_root.next_state[i];
    }

    int tape[2 * half_tape_length + 1] = {0};
    int head_position = half_tape_length;
    int current_state = 0;
    int steps = 0;
    int depth = 0;

    // for some reason that is far beyond me, tm.num_symbols resets at the end of each iteration of the while loop, but this gets around that
    int horrible_thing_why_does_this_work = tm.num_symbols;
    while (steps < max_steps) {
        tm.num_symbols = horrible_thing_why_does_this_work;

        // the transitions are indexed like this (but continued up to CODON_MEM_LIMIT-1):
        // 0  2  5  10
        // 1  3  7  12
        // 4  6  8  14
        // 9  11 13 15
        int action_layer = max(current_state, tape[head_position]) - 1;
        int action_side = (int)(current_state < tape[head_position]);
        int action_index = action_layer * action_layer + 2 * (current_state + tape[head_position]) + action_side - 1;
        if (action_index >= CODON_MEM_LIMIT) {
            break;
        }

        current_state = tm.next_state[action_index];
        if(current_state == -1) {
            // borders
            Cuda::vec2 corner_shift = Cuda::vec2(border_thickness, border_thickness) * (rx_by - lx_ty);
            lx_ty += corner_shift;
            rx_by -= corner_shift;
            if (!(lx_ty.x < idx && idx < rx_by.x && lx_ty.y < idy && idy < rx_by.y)) {
                tape[head_position] = tm.write_symbol[action_index];
                break;
            }

            get_child(tm, action_index, Cuda::vec2(idx, idy), lx_ty, rx_by);
            depth += 1;
        }
        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        current_state = tm.next_state[action_index];
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
        horrible_thing_why_does_this_work = tm.num_symbols;
    }
    /*int num_ones = 0;
    for (int i = 0; i < 2 * half_tape_length + 1; i++) {
        if (tape[i] == 1) {
            num_ones++;
        }
    }*/

    //bool halted = current_state == -1;

    //int col = 0xff000000 | (0x3f << ((head_position - 2 * half_tape_length + 2) * 4));
    //int col = 0xff000000 | (0x3f << (current_state * 8));
    //pixels[pixel_index] = halted ? d_rainbow((float) head_position / 42) : 0xff000000;
    //float atan_steps = atanf(steps / 4.) / 1.57079632679f; // Normalize to [0, 1]
    //pixels[pixel_index] = halted ? d_rainbow(atan_steps) : 0xff000000;
    //pixels[pixel_index] = 0xff000000 + 0x00111111 * min(depth, 15);
    depth += 2;
    pixels[pixel_index] = d_rainbow(atanf(depth*depth / 200.) / 1.57079632679f);
}

extern "C" void beaver_grid_TNF_cuda(Color* pixels, int w, int h, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, int max_steps) {
    Color* d_pixels;
    size_t size = w * h * sizeof(Color);
    cudaMalloc(&d_pixels, size);
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    // for now we're only doing empty TNF root, since you can get everything else by zooming in correctly anyway
    TuringMachine TNF_root;
    TNF_root.num_states = 2;
    TNF_root.num_symbols = 2;
    for (int i = 0; i < CODON_MEM_LIMIT; i++) {
        TNF_root.write_symbol[i] = 0;
        TNF_root.left_right[i] = true;
        TNF_root.next_state[i] = -1;
    }

    beaver_TNF_kernel<<<gridSize, blockSize>>>(d_pixels, w, h, lx_ty, rx_by, true, 0.02, TNF_root, max_steps);

    cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
