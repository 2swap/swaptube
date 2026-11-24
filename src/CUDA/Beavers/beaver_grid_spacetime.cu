#include <cuda_runtime.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/Color.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../color.cuh"
#include "../common_graphics.cuh"

// Picks the 2-state machine for a grid cell. The grid enumerates the four
// transitions in the order (A,0), (B,0), (A,1), (B,1): slot k of the local
// arrays below is the transition at codon order_to_codon[k].
__device__ void get_TM_from_pos(Cuda::ivec2 cell_pos, Cuda::TuringMachine& tm) {
    tm.num_states = 2;
    const int order_to_codon[4] = {Cuda::tm_codon(0, 0), Cuda::tm_codon(1, 0), Cuda::tm_codon(0, 1), Cuda::tm_codon(1, 1)};
    int next_state[4], write_symbol[4], move_right[4];
    int pows[4] = {1,3,9,27};
    for (int i=0; i<4; i++) {
        int val = 3 * ((cell_pos.y / pows[3 - i]) % 3) + ((cell_pos.x / pows[3 - i]) % 3);
        int j = i ^ (i > 1 && next_state[write_symbol[0]] != 0);
        write_symbol[j] = val & 1;
        move_right[j] = (val & 2) == 2;
        next_state[j] = (val >> 2) - 3 * (val == 8);   // -1 means halt
    }
    for (int k=0; k<4; k++) {
        tm.table[order_to_codon[k]] = Cuda::tm_pack(next_state[k] == -1 ? Cuda::TM_HALT : next_state[k], write_symbol[k], move_right[k]);
    }
}

__device__ uint32_t get_color_at_cell(Cuda::ivec2 grid_cell_pos, Cuda::ivec2 spacetime_cell_pos, Cuda::vec2 grid_wh, Cuda::vec2 spacetime_wh, int iterations) {
    bool inside_grid = grid_cell_pos.x >= 0 && grid_cell_pos.y >= 0 && grid_cell_pos.x < grid_wh.x && grid_cell_pos.y < grid_wh.y;
    bool inside_spacetime = spacetime_cell_pos.x >= 0 && spacetime_cell_pos.y >= 0 && spacetime_cell_pos.x < spacetime_wh.x && spacetime_cell_pos.y < spacetime_wh.y;

    if (!inside_grid) return 0x00000000;

    Cuda::TuringMachine tm;
    get_TM_from_pos(grid_cell_pos, tm);

    uint32_t colors[3] = {0xff0050ff, 0xff805800, 0xffff8078};

    // run TM
    const int half_tape_length = 10;
    uint8_t tape[2 * half_tape_length + 1] = {0};
    int head_position = half_tape_length;
    int current_state = 0;
    int steps = 0;
    bool halted = false;
    uint32_t cell_color = 0xff000000;
    while (steps < iterations) {
        if (steps == spacetime_cell_pos.y && inside_spacetime) cell_color = colors[2 - (head_position != spacetime_cell_pos.x+5) * (2 - tape[spacetime_cell_pos.x+5])];
        int tr = tm.table[Cuda::tm_codon(current_state, tape[head_position])];
        if (Cuda::tm_next_state(tr) == Cuda::TM_HALT) {
            halted = true;
            break;
        }
        current_state = Cuda::tm_next_state(tr);
        tape[head_position] = Cuda::tm_write_symbol(tr);
        head_position += 2 * Cuda::tm_move_right(tr) - 1;
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
    }
    return cell_color + (halted && !inside_spacetime) * 0x0000ff00;
}

/*__device__ Cuda::ivec2 clamp(Cuda::ivec2 v, Cuda::ivec2 l, Cuda::ivec2 u) {
    return Cuda::ivec2(min(u.x, max(l.x, v.x)), min(u.y, max(l.y, v.y)));
}*/

__global__ void grid_spacetime_kernel(
    uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    Cuda::vec2 grid_wh, Cuda::vec2 spacetime_wh, float tm_border, int iterations
) {
    Cuda::ivec2 pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= wh.x || pos.y >= wh.y /*|| (pos.x & 7) != 0 || (pos.y & 7) != 0*/) {
        return;
    }
    int pixel_index = pos.y * wh.x + pos.x;

    Cuda::vec2 point_vec_tl = pos * (rx_by - lx_ty) / wh + lx_ty;
    Cuda::ivec2 grid_cell_pos_tl = -floor(-clamp(floor(point_vec_tl * grid_wh), Cuda::ivec2(-1), grid_wh));
    Cuda::vec2 grid_pos_decimal_tl = point_vec_tl * grid_wh - grid_cell_pos_tl;
    Cuda::vec2 spacetime_pos_tl = (grid_pos_decimal_tl - 0.5f) / (1 - tm_border) + 0.5f;
    Cuda::ivec2 spacetime_cell_pos_tl = -floor(-clamp(floor(spacetime_pos_tl * spacetime_wh), Cuda::ivec2(-1), spacetime_wh));
    Cuda::vec2 spacetime_pos_decimal_tl = spacetime_pos_tl * spacetime_wh - spacetime_cell_pos_tl;

    /*Cuda::vec2 point_vec_br = (pos + 1) * (rx_by - lx_ty) / wh + lx_ty;
    Cuda::ivec2 grid_cell_pos_br = ceil(clamp(floor(point_vec_br * grid_wh), Cuda::ivec2(-1), grid_wh));
    Cuda::vec2 grid_pos_decimal_br = point_vec_br * grid_wh - grid_cell_pos_br;
    Cuda::vec2 spacetime_pos_br = (grid_pos_decimal_br - 0.5f) / (1 - tm_border) + 0.5f;
    Cuda::ivec2 spacetime_cell_pos_br = ceil(clamp(floor(spacetime_pos_br * spacetime_wh), Cuda::ivec2(-1), spacetime_wh));
    Cuda::vec2 spacetime_pos_decimal_br = spacetime_pos_br * spacetime_wh - spacetime_cell_pos_br;

    Cuda::ivec2 x(1,0);
    Cuda::ivec2 y(0,1);
    float d_l = (spacetime_cell_pos_tl.x < spacetime_wh.x) * (1 - spacetime_pos_decimal_tl.x + max(0, -1 - spacetime_cell_pos_tl.x)) + (spacetime_cell_pos_tl.x >= spacetime_wh.x) * (1 - grid_pos_decimal_tl.x) / (1 - tm_border) * spacetime_wh.x;
    float d_r = (spacetime_cell_pos_br.x >= 0) * (spacetime_pos_decimal_br.x + max(0, spacetime_cell_pos_tl.x - spacetime_wh.x)) + (spacetime_cell_pos_tl.x < 0) * grid_pos_decimal_br.x / (1 - tm_border) * spacetime_wh.x;
    float d_t = (spacetime_cell_pos_tl.y < spacetime_wh.y) * (1 - spacetime_pos_decimal_tl.y + max(0, -1 - spacetime_cell_pos_tl.y)) + (spacetime_cell_pos_tl.y >= spacetime_wh.y) * (1 - grid_pos_decimal_tl.y) / (1 - tm_border) * spacetime_wh.y;
    float d_b = (spacetime_cell_pos_br.y >= 0) * (spacetime_pos_decimal_br.y + max(0, spacetime_cell_pos_tl.y - spacetime_wh.y)) + (spacetime_cell_pos_tl.y < 0) * grid_pos_decimal_br.y / (1 - tm_border) * spacetime_wh.y;
    float w_x = d_l / (d_l + d_r);
    float w_y = d_t / (d_t + d_b);*/

    uint32_t col_tl = get_color_at_cell(grid_cell_pos_tl, spacetime_cell_pos_tl, grid_wh, spacetime_wh, iterations);
    /*uint32_t col_bl = get_color_at_cell(grid_cell_pos_tl*x+grid_cell_pos_br*y, spacetime_cell_pos_tl*x+spacetime_cell_pos_br*y, grid_wh, spacetime_wh, iterations);
    uint32_t col_tr = get_color_at_cell(grid_cell_pos_tl*y+grid_cell_pos_br*x, spacetime_cell_pos_tl*y+spacetime_cell_pos_br*x, grid_wh, spacetime_wh, iterations);
    uint32_t col_br = get_color_at_cell(grid_cell_pos_br, spacetime_cell_pos_br, grid_wh, spacetime_wh, iterations);
    uint32_t col = Cuda::colorlerp(
        Cuda::colorlerp(col_tl, col_bl, 1-w_y),
        Cuda::colorlerp(col_tr, col_br, 1-w_y),
        1-w_x
    );*/

    pixels[pixel_index] = col_tl;

    //if (pos.y == 
    /*if (pos.y == 67 && pos.x == 80) {
        printf("\n%f  %f  %f  %f  %f  (%d,%d,%d,%d)", point_vec_tl.y * grid_wh.y, spacetime_pos_tl.y * spacetime_wh.y, spacetime_pos_decimal_tl.y, spacetime_pos_decimal_br.y, w_y, col>>24, (col>>16)&255, (col>>8)&255, col&255);
    }*/
}

extern "C" void beaver_grid_spacetime(
    uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    Cuda::vec2 grid_wh, Cuda::vec2 spacetime_wh, float tm_border, float iterations
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    grid_spacetime_kernel<<<gridSize, blockSize>>>(
        pixels, wh, lx_ty, rx_by,
        grid_wh, spacetime_wh, tm_border, iterations
    );
}