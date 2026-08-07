#include <cuda_runtime.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/Color.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../color.cuh"
#include "../common_graphics.cuh"

__device__ void get_TM_from_pos(Cuda::ivec2 cell_pos, TuringMachine& tm) {
    tm.num_states = 2;
    tm.num_symbols = 2;
    int pows[4] = {1,3,9,27};
    for (int i=0; i<4; i++) {
        int val = 3 * ((cell_pos.y / pows[3 - i]) % 3) + ((cell_pos.x / pows[3 - i]) % 3);
        int j = i ^ (i > 1 && tm.next_state[tm.write_symbol[0]]);
        tm.write_symbol[j] = val & 1;
        tm.left_right[j] = (val & 2) == 2;
        tm.next_state[j] = (val >> 2) - 3 * (val == 8);
    }
}

__device__ uint32_t get_color_at_cell(Cuda::ivec2 grid_cell_pos, Cuda::ivec2 spacetime_cell_pos, Cuda::ivec2 grid_wh, Cuda::ivec2 spacetime_wh, int iterations) {
    bool inside_grid = grid_cell_pos.x >= 0 && grid_cell_pos.y >= 0 && grid_cell_pos.x < grid_wh.x && grid_cell_pos.y < grid_wh.y;
    bool inside_spacetime = spacetime_cell_pos.x >= 0 && spacetime_cell_pos.y >= 0 && spacetime_cell_pos.x < spacetime_wh.x && spacetime_cell_pos.y < spacetime_wh.y;

    if (!inside_grid) return 0x00000000;

    TuringMachine tm;
    get_TM_from_pos(grid_cell_pos, tm);

    uint32_t colors[3] = {0xff0050ff, 0xff805800, 0xffff8078};

    // run TM
    const int half_tape_length = 10;
    int tape[2 * half_tape_length + 1] = {0};
    int head_position = half_tape_length;
    int current_state = 0;
    int steps = 0;
    int action_index;
    uint32_t cell_color = 0xff000000;
    while (steps < iterations) {
        if (steps == spacetime_cell_pos.y && inside_spacetime) cell_color = colors[2 - (head_position != spacetime_cell_pos.x+5) * (2 - tape[spacetime_cell_pos.x+5])];
        // the transitions are indexed like this (but continued up to CODON_MEM_LIMIT-1):
        // 0  2  5  10
        // 1  3  7  12
        // 4  6  8  14
        // 9  11 13 15
        int action_layer = max(current_state, tape[head_position]) - 1;
        int action_side = (int)(current_state < tape[head_position]);
        action_index = action_layer * action_layer + 2 * (current_state + tape[head_position]) + action_side - 1;
        if (action_index >= CODON_MEM_LIMIT) {
            break;
        }

        current_state = tm.next_state[action_index];
        if (current_state == -1) {
            break;
        }
        tape[head_position] = tm.write_symbol[action_index];
        head_position += 2 * tm.left_right[action_index] - 1;
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
    }
    bool halted = current_state == -1;

    return cell_color + (halted && !inside_spacetime) * 0x0000ff00;
}

__global__ void grid_spacetime_kernel(
    uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    Cuda::ivec2 grid_wh, Cuda::ivec2 spacetime_wh, float tm_border, int iterations
) {
    Cuda::ivec2 pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= wh.x || pos.y >= wh.y) {
        return;
    }
    int pixel_index = pos.y * wh.x + pos.x;

    Cuda::vec2 point_vec_tl = pos * (rx_by - lx_ty) / wh + lx_ty;
    Cuda::ivec2 grid_cell_pos_tl = floor(point_vec_tl * grid_wh);
    Cuda::vec2 grid_pos_decimal_tl = point_vec_tl * grid_wh - grid_cell_pos_tl;
    Cuda::vec2 spacetime_pos_tl = (grid_pos_decimal_tl - 0.5f) / (1 - tm_border) + 0.5f;
    Cuda::ivec2 spacetime_cell_pos_tl = floor(spacetime_pos_tl * spacetime_wh);
    Cuda::vec2 spacetime_pos_decimal_tl = spacetime_pos_tl * spacetime_wh - spacetime_cell_pos_tl;

    Cuda::vec2 point_vec_br = (pos + 1) * (rx_by - lx_ty) / wh + lx_ty;
    Cuda::ivec2 grid_cell_pos_br = floor(point_vec_br * grid_wh);
    Cuda::vec2 grid_pos_decimal_br = point_vec_br * grid_wh - grid_cell_pos_br;
    Cuda::vec2 spacetime_pos_br = (grid_pos_decimal_br - 0.5f) / (1 - tm_border) + 0.5f;
    Cuda::ivec2 spacetime_cell_pos_br = floor(spacetime_pos_br * spacetime_wh);
    Cuda::vec2 spacetime_pos_decimal_br = spacetime_pos_br * spacetime_wh - spacetime_cell_pos_br;

    Cuda::ivec2 x(1,0);
    Cuda::ivec2 y(0,1);
    float w_x = (1 - spacetime_pos_decimal_tl.x) / (1 - spacetime_pos_decimal_tl.x + spacetime_pos_decimal_br.x);
    float w_y = (1 - spacetime_pos_decimal_tl.y) / (1 - spacetime_pos_decimal_tl.y + spacetime_pos_decimal_br.y);
    pixels[pixel_index] = Cuda::colorlerp(
        Cuda::colorlerp(
            get_color_at_cell(grid_cell_pos_tl, spacetime_cell_pos_tl, grid_wh, spacetime_wh, iterations),
            get_color_at_cell(grid_cell_pos_tl*x+grid_cell_pos_br*y, spacetime_cell_pos_tl*x+spacetime_cell_pos_br*y, grid_wh, spacetime_wh, iterations),
            w_y
        ),
        Cuda::colorlerp(
            get_color_at_cell(grid_cell_pos_tl*y+grid_cell_pos_br*x, spacetime_cell_pos_tl*y+spacetime_cell_pos_br*x, grid_wh, spacetime_wh, iterations),
            get_color_at_cell(grid_cell_pos_br, spacetime_cell_pos_br, grid_wh, spacetime_wh, iterations),
            w_y
        ),
        w_x
    );
}

extern "C" void beaver_grid_spacetime(
    uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    Cuda::ivec2 grid_wh, Cuda::ivec2 spacetime_wh, float tm_border, float iterations
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    grid_spacetime_kernel<<<gridSize, blockSize>>>(
        pixels, wh, lx_ty, rx_by,
        grid_wh, spacetime_wh, tm_border, iterations
    );
}