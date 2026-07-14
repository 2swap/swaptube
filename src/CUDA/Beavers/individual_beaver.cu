#include <cuda_runtime.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/Color.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../color.cuh"
#include "../common_graphics.cuh"

__device__ uint32_t opacity_multiply(uint32_t color, float opacity) {
    return Cuda::colorlerp(color & 0x00ffffff, color, opacity);
}

__device__ Cuda::vec2 scaleify(Cuda::vec2 thing) {
    return thing / fminf(thing.x, thing.y);
}

__global__ void individual_beaver_kernel(
    uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    uint32_t* grid, Cuda::ivec2 grid_wh,
    uint32_t* icons, Cuda::ivec2 icons_wh, int icons_len,
    TuringMachine tm, float iterations,
    float state_icon_scale, float vertical_step, float opacity_min, float opacity_dropoff,
    float dir_icon_scale, float current_tape_opacity, int rest,
    Cuda::vec2 table_wh, Cuda::vec2 table_wh0, float table_margin, float icon_border, float table_border, float table_glow
) {
    Cuda::ivec2 pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= wh.x || pos.y >= wh.y) {
        return;
    }
    int pixel_index = pos.y * wh.x + pos.x;
    pixels[pixel_index] = 0x00000000;



    // spacetime diagram

    Cuda::vec2 point_vec = pos * (rx_by - lx_ty) / wh + lx_ty;
    Cuda::vec2 grid_pos = point_vec + Cuda::vec2(grid_wh.x / 2.0f, 0);
    Cuda::ivec2 spacetime_topmost_cell_pos = Cuda::ivec2(int(grid_pos.x), int(fminf(grid_wh.y, grid_pos.y / vertical_step)));

    for (int t=int(fminf(grid_wh.y, fmaxf(0, (grid_pos.y-1) / vertical_step + 1))); t<=spacetime_topmost_cell_pos.y; t++) {
        float opacity = opacity_dropoff < 1 ? fmaxf(opacity_min, pow(opacity_dropoff, iterations - t)) : fminf(1, opacity_min * pow(opacity_dropoff, iterations - t));
        Cuda::ivec2 cell_pos = Cuda::ivec2(spacetime_topmost_cell_pos.x, t);
        Cuda::vec2 grid_pos_decimal = grid_pos - cell_pos * Cuda::vec2(1, vertical_step);

        if (cell_pos.x >= 0 && cell_pos.y >= 0 && cell_pos.x < grid_wh.x && cell_pos.y < grid_wh.y - 1) {
            uint32_t cell = grid[cell_pos.y * grid_wh.x + cell_pos.x];

            int symbol_icon = cell >> 16;
            Cuda::ivec2 symbol_pos = Cuda::ivec2(int(grid_pos_decimal.x * icons_wh.x), int(grid_pos_decimal.y * icons_wh.y));
            if (symbol_icon < icons_len && symbol_pos.y < icons_wh.y) {
                uint32_t symbol_pixel = icons[(symbol_icon * icons_wh.y + symbol_pos.y) * icons_wh.x + symbol_pos.x];
                pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], opacity_multiply(symbol_pixel, opacity));
            }

            int state_icon = (cell & 0x0000ffff) + tm.num_symbols;
            if (state_icon < icons_len && state_icon_scale > 0) {
                float t = 1 / state_icon_scale;
                Cuda::ivec2 state_pos = Cuda::ivec2(int(floor((grid_pos_decimal.x * t + 0.5f * (1 - t)) * icons_wh.x)), int(floor((grid_pos_decimal.y * t + 0.5f * (1 - t)) * icons_wh.y)));
                if (state_pos.x >= 0 && state_pos.y >= 0 && state_pos.x < icons_wh.x && state_pos.y < icons_wh.y) {
                    uint32_t state_pixel = icons[(state_icon * icons_wh.y + state_pos.y) * icons_wh.x + state_pos.x];
                    pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], opacity_multiply(state_pixel, opacity));
                }
            }
        }
    }



    // current tape

    Cuda::vec2 cur_tape_pos = grid_pos - Cuda::vec2(0, iterations * vertical_step);

    if (cur_tape_pos.x >= 0 && cur_tape_pos.y >= 0 && cur_tape_pos.x <= grid_wh.x && cur_tape_pos.y <= 1) {
        Cuda::ivec2 cell_pos = spacetime_topmost_cell_pos;
        Cuda::vec2 grid_pos_decimal = grid_pos - cell_pos * Cuda::vec2(1, vertical_step);
        uint32_t cell0 = grid[max(0, grid_wh.y-2) * grid_wh.x + cell_pos.x];
        uint32_t cell1 = grid[(grid_wh.y-1) * grid_wh.x + cell_pos.x];

        int symbol_icon0 = cell0 >> 16;
        int symbol_icon1 = cell1 >> 16;
        Cuda::ivec2 symbol_pos = Cuda::ivec2(int(grid_pos_decimal.x * icons_wh.x), int(cur_tape_pos.y * icons_wh.y));
        if (symbol_icon0 >= icons_len) symbol_icon0 = symbol_icon1;
        if (symbol_icon1 >= icons_len) symbol_icon1 = symbol_icon0;
        if (symbol_icon0 < icons_len && symbol_pos.x >= 0 && symbol_pos.y >= 0 && symbol_pos.x < icons_wh.x && symbol_pos.y < icons_wh.y) {
            uint32_t symbol_pixel0 = icons[(symbol_icon0 * icons_wh.y + symbol_pos.y) * icons_wh.x + symbol_pos.x];
            uint32_t symbol_pixel1 = icons[(symbol_icon1 * icons_wh.y + symbol_pos.y) * icons_wh.x + symbol_pos.x];
            uint32_t symbol_pixel = Cuda::colorlerp(symbol_pixel0, symbol_pixel1, iterations-min(grid_wh.y-2, int(iterations)));
            pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], opacity_multiply(symbol_pixel, current_tape_opacity));
        }

        if (dir_icon_scale > 0 && (cell0 & cell1 & 0x0000ffff) != 0x0000ffff) {
            uint32_t cell2 = grid[max(0, grid_wh.y-1-((cell0 & 0x0000ffff) == 0x0000ffff)) * grid_wh.x + max(0,cell_pos.x-1)];
            uint32_t cell3 = grid[max(0, grid_wh.y-1-((cell0 & 0x0000ffff) == 0x0000ffff)) * grid_wh.x + min(grid_wh.x-1,cell_pos.x+1)];
            int dir_sign = (((cell2 & 0x0000ffff) == 0x0000ffff) - ((cell3 & 0x0000ffff) == 0x0000ffff)) * (1 - 2 * ((cell0 & 0x0000ffff) == 0x0000ffff));
            int dir_icon = ((dir_sign+1)/2) + tm.num_states + tm.num_symbols;
            dir_icon += (rest != 0) * (tm.num_states + tm.num_symbols + rest + 1 - dir_icon);

            Cuda::vec2 dir_pos_raw = grid_pos - Cuda::vec2(cell_pos.x - (((cell0 & 0x0000ffff) == 0x0000ffff) - Cuda::smoother2(iterations - min(grid_wh.y-2, int(iterations)))) * dir_sign, iterations * vertical_step);
            if (dir_pos_raw.x >= 0 && dir_pos_raw.x < 1) {
                float t = 1 / dir_icon_scale;
                Cuda::ivec2 dir_pos = Cuda::ivec2(int(floor((dir_pos_raw.x * t + 0.5f * (1 - t)) * icons_wh.x)), int(floor((cur_tape_pos.y * t + 0.5f * (1 - t)) * icons_wh.y)));
                if (dir_pos.x >= 0 && dir_pos.y >= 0 && dir_pos.x < icons_wh.x && dir_pos.y < icons_wh.y) {
                    uint32_t dir_pixel = dir_icon < icons_len ? icons[(dir_icon * icons_wh.y + dir_pos.y) * icons_wh.x + dir_pos.x] : 0x00000000;
                    pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], opacity_multiply(dir_pixel, current_tape_opacity));
                }
            }
        }
    }



    // transition table

    if (table_wh.x > 0 && table_wh.y > 0) {
        float shorter_side_pixel_length = fminf(table_wh.x * wh.x, table_wh.y * wh.y);
        Cuda::vec2 scale = table_wh * wh / shorter_side_pixel_length;
        Cuda::vec2 table_pos = (Cuda::vec2(pos) / wh - (1 - Cuda::vec2(table_border * wh.y / wh.x, table_border))) / table_wh + Cuda::ivec2(tm.num_symbols, tm.num_states);
        Cuda::ivec2 cell_pos(floor(table_pos));
        cell_pos = Cuda::ivec2(min(tm.num_symbols, max(-1, cell_pos.x)), min(tm.num_states, max(-1, cell_pos.y)));
        Cuda::vec2 table_pos_decimal = table_pos - cell_pos;
        Cuda::vec2 table_cell_size(1 + (cell_pos.x < 0) * (table_wh0.x - 1), 1 + (cell_pos.y < 0) * (table_wh0.y - 1));
        cell_pos -= Cuda::ivec2(table_pos_decimal.x < 1 - table_cell_size.x, table_pos_decimal.y < 1 - table_cell_size.y);
        table_pos_decimal = 1 - (1 - table_pos_decimal) / table_cell_size;
        bool inside_table = min(0, cell_pos.x) + min(0, cell_pos.y) >= -1 && cell_pos.x < tm.num_symbols && cell_pos.y < tm.num_states;

        // lines
        Cuda::vec4 glow_wnes(
            fminf(1, fmaxf(0, inside_table * (1 - table_pos_decimal.x * table_cell_size.x * scale.x / table_glow))),
            fminf(1, fmaxf(0, inside_table * (1 - table_pos_decimal.y * table_cell_size.y * scale.y / table_glow))),
            fminf(1, fmaxf(0, inside_table * (1 - (1 - table_pos_decimal.x) * table_cell_size.x * scale.x / table_glow))),
            fminf(1, fmaxf(0, inside_table * (1 - (1 - table_pos_decimal.y) * table_cell_size.y * scale.y / table_glow)))
        );
        float max_glow = fmaxf(glow_wnes.x, fmaxf(glow_wnes.y, fmaxf(glow_wnes.z, glow_wnes.w)));
        float glow = fmaxf(1 - (1 - glow_wnes.x) * (1 - glow_wnes.y) * (1 - glow_wnes.z) * (1 - glow_wnes.w), (1 - max_glow) * table_glow * shorter_side_pixel_length < 1);
        uint32_t glow_color = opacity_multiply(Cuda::black_to_blue_to_white(glow), glow);
        pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], glow_color);

        // icons
        if (inside_table && table_margin < 0.5f) {
            bool is_transition = cell_pos.x >= 0 && cell_pos.y >= 0;
            Cuda::vec2 margin_pos = (table_pos_decimal - table_margin) / (1 - 2 * table_margin);
            float content_aspect_ratio = 1 + is_transition * 2 * (1 + icon_border);
            Cuda::vec2 content_pos = 0.5f + scaleify(Cuda::vec2(1, content_aspect_ratio)) * scale * scaleify(table_cell_size) * (margin_pos - 0.5f);
            int icon_cell = int(content_pos.x * content_aspect_ratio / (1 + icon_border));
            Cuda::ivec2 icon_pos(floor((content_pos * Cuda::vec2(content_aspect_ratio, 1) - Cuda::vec2(icon_cell * (1 + icon_border), 0)) * icons_wh));
            if (content_pos.x >= 0 && content_pos.y >= 0 && content_pos.x < 1 && content_pos.y < 1 && icon_pos.x >= 0 && icon_pos.y >= 0 && icon_pos.x < icons_wh.x && icon_pos.y < icons_wh.y) {
                int action_layer = max(cell_pos.x, cell_pos.y) - 1;
                int action_side = cell_pos.x > cell_pos.y;
                int action_index = max(0, action_layer * action_layer + 2 * (cell_pos.x + cell_pos.y) + action_side - 1);
                int transition[3] = {tm.write_symbol[action_index], tm.left_right[action_index] + tm.num_symbols + tm.num_states, tm.next_state[action_index] + tm.num_symbols};
                int icon = is_transition * transition[icon_cell] + (!is_transition) * ((cell_pos.y == -1) * cell_pos.x + (cell_pos.x == -1) * (cell_pos.y + tm.num_symbols));
                int icon_pixel = (icon >= 0 && icon < icons_len) ? icons[(icon * icons_wh.y + icon_pos.y) * icons_wh.x + icon_pos.x] : 0x00000000;
                pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], icon_pixel);
            }
        }
    }
}

extern "C" void draw_individual_beaver(
    uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    uint32_t* grid, Cuda::ivec2 grid_wh,
    uint32_t* icons, Cuda::ivec2 icons_wh, int icons_len,
    TuringMachine tm, float iterations,
    float state_icon_scale, float vertical_step, float opacity_min, float opacity_dropoff,
    float dir_icon_scale, float current_tape_opacity, int rest,
    Cuda::vec2 table_wh, Cuda::vec2 table_wh0, float table_margin, float icon_border, float table_border, float table_glow
) {
    uint32_t* d_grid;
    size_t grid_size = grid_wh.x * grid_wh.y * sizeof(uint32_t);
    cudaMalloc(&d_grid, grid_size);
    cudaMemcpy(d_grid, grid, grid_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    individual_beaver_kernel<<<gridSize, blockSize>>>(
        pixels, wh, lx_ty, rx_by,
        d_grid, grid_wh,
        icons, icons_wh, icons_len,
        tm, iterations,
        state_icon_scale, vertical_step, opacity_min, opacity_dropoff,
        dir_icon_scale, current_tape_opacity, rest,
        table_wh, table_wh0, table_margin, icon_border, table_border, table_glow
    );

    cudaFree(d_grid);
}