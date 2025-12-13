#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <utility>
#include <glm/glm.hpp>
#include "../Host_Device_Shared/helpers.h"

typedef uint64_t Bitboard;

const Bitboard ULB = 0x007f7f7f7f7f7f7f;
const Bitboard ULC = 0x8000000000000000;
const Bitboard ULS = 0x7f00000000000000;
const Bitboard ULZ = 0x0080808080808080;

const Bitboard URB = 0x00fefefefefefefe;
const Bitboard URC = 0x0100000000000000;
const Bitboard URS = 0x0001010101010101;
const Bitboard URZ = 0xfe00000000000000;

const Bitboard DLB = 0x7f7f7f7f7f7f7f00;
const Bitboard DLC = 0x0000000000000080;
const Bitboard DLS = 0x8080808080808000;
const Bitboard DLZ = 0x000000000000007f;

const Bitboard DRB = 0xfefefefefefefe00;
const Bitboard DRC = 0x0000000000000001;
const Bitboard DRZ = 0x0101010101010100;
const Bitboard DRS = 0x00000000000000fe;

const Bitboard RW1 = 0x0101010101010101;
const Bitboard DW1 = 0x00000000000000ff;
const Bitboard UW1 = 0xff00000000000000;
const Bitboard LW1 = 0x8080808080808080;

const Bitboard RW7 = 0x7f7f7f7f7f7f7f7f;
const Bitboard DW7 = 0x00ffffffffffffff;
const Bitboard UW7 = 0xffffffffffffff00;
const Bitboard LW7 = 0xfefefefefefefefe;

__global__ void conway_kernel(Bitboard* board, Bitboard* board_2, int w_bitboards, int h_bitboards) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= w_bitboards * h_bitboards) return;

    int bx = idx % w_bitboards;
    int by = idx / w_bitboards;
    int d = (by + 1) % h_bitboards;
    int l = (bx + 1) % w_bitboards;
    int u = (by + h_bitboards - 1) % h_bitboards;
    int r = (bx + w_bitboards - 1) % w_bitboards;

    Bitboard ul = board[u * w_bitboards + l];
    Bitboard uc = board[u * w_bitboards + bx];
    Bitboard ur = board[u * w_bitboards + r];

    Bitboard cl = board[by * w_bitboards + l];
    Bitboard cc = board[idx];
    Bitboard cr = board[by * w_bitboards + r];

    Bitboard dl = board[d * w_bitboards + l];
    Bitboard dc = board[d * w_bitboards + bx];
    Bitboard dr = board[d * w_bitboards + r];

    Bitboard sul = ((ul << 63) & ULC) | ((cc >> 9) & ULB) | ((uc << 55) & ULS) | ((cl >> 01) & ULZ);
    Bitboard sur = ((ur << 49) & URC) | ((cc >> 7) & URB) | ((uc << 57) & URZ) | ((cr >> 15) & URS);
    Bitboard sdl = ((dl >> 49) & DLC) | ((cc << 7) & DLB) | ((dc >> 57) & DLZ) | ((cl << 15) & DLS);
    Bitboard sdr = ((dr >> 63) & DRC) | ((cc << 9) & DRB) | ((dc >> 55) & DRS) | ((cr << 01) & DRZ);

    Bitboard suc = ((uc << 56) & UW1) | ((cc >> 8) & DW7);
    Bitboard scl = ((cl << 07) & LW1) | ((cc >> 1) & RW7);
    Bitboard scr = ((cr >> 07) & RW1) | ((cc << 1) & LW7);
    Bitboard sdc = ((dc >> 56) & DW1) | ((cc << 8) & UW7);
    
    // Half adders
    Bitboard ones_sul_suc = sul ^ suc;
    Bitboard twos_sul_suc = sul & suc;

    Bitboard ones_sur_scl = sur ^ scl;
    Bitboard twos_sur_scl = sur & scl;

    Bitboard ones_sdl_sdc = sdl ^ sdc;
    Bitboard twos_sdl_sdc = sdl & sdc;

    Bitboard ones_sdr_scr = sdr ^ scr;
    Bitboard twos_sdr_scr = sdr & scr;

    // Sum the intermediate results with full adders
    Bitboard top4_ones = ones_sul_suc ^ ones_sur_scl;
    Bitboard top4_ones_carry = ones_sul_suc & ones_sur_scl;
    Bitboard top4_twos = twos_sul_suc ^ twos_sur_scl ^ top4_ones_carry;
    Bitboard top4_twos_carry = (twos_sul_suc & twos_sur_scl) | (top4_ones_carry & (twos_sul_suc ^ twos_sur_scl));
    Bitboard top4_fours = top4_twos_carry;

    Bitboard bottom4_ones = ones_sdl_sdc ^ ones_sdr_scr;
    Bitboard bottom4_ones_carry = ones_sdl_sdc & ones_sdr_scr;
    Bitboard bottom4_twos = twos_sdl_sdc ^ twos_sdr_scr ^ bottom4_ones_carry;
    Bitboard bottom4_twos_carry = (twos_sdl_sdc & twos_sdr_scr) | (bottom4_ones_carry & (twos_sdl_sdc ^ twos_sdr_scr));
    Bitboard bottom4_fours = bottom4_twos_carry;

    // Final sum
    Bitboard final_ones = top4_ones ^ bottom4_ones;
    Bitboard final_ones_carry = top4_ones & bottom4_ones;
    Bitboard final_twos = top4_twos ^ bottom4_twos ^ final_ones_carry;
    Bitboard final_twos_carry = (top4_twos & bottom4_twos) | (final_ones_carry & (top4_twos ^ bottom4_twos));
    Bitboard final_fours = top4_fours ^ bottom4_fours ^ final_twos_carry;
    Bitboard final_fours_carry = (top4_fours & bottom4_fours) | (final_twos_carry & (top4_fours ^ bottom4_fours));
    Bitboard final_eights = final_fours_carry;

    // Apply Conway's rules
    Bitboard is_three_neighbors = final_ones & final_twos & ~final_fours & ~final_eights;
    Bitboard is_two_neighbors = ~final_ones & final_twos & ~final_fours & ~final_eights;
    Bitboard next_state = (cc & is_two_neighbors) | is_three_neighbors;
    board_2[idx] = next_state;
}

extern "C" void iterate_conway(Bitboard* d_board, Bitboard* d_board_2, int w_bitboards, int h_bitboards)
{
    dim3 blockSize(256);
    dim3 numBlocks((w_bitboards * h_bitboards + blockSize.x - 1) / blockSize.x);
    conway_kernel<<<numBlocks, blockSize>>>(d_board, d_board_2, w_bitboards, h_bitboards);
    cudaDeviceSynchronize();
}

__global__ void conway_draw_kernel(Bitboard* board, int w_bitboards, int h_bitboards, unsigned int* pixels, int pixels_w, int pixels_h, glm::vec2 lx_ty, glm::vec2 rx_by)
{
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px >= pixels_w || py >= pixels_h) return;

    int pixel_idx = py * pixels_w + px;
    glm::vec2 point_vec = pixel_to_point(glm::vec2(px, py), lx_ty, rx_by, glm::vec2(pixels_w, pixels_h));
    point_vec += glm::vec2(4.0f * w_bitboards, 4.0f * h_bitboards); // Centering

    if(point_vec.x < 0 || point_vec.y < 0) {
        pixels[pixel_idx] = 0xff808080;
        return;
    }
    if(point_vec.x >= w_bitboards * 8 || point_vec.y >= h_bitboards * 8) {
        pixels[pixel_idx] = 0xff808080;
        return;
    }

    int board_x = (int)(point_vec.x) / 8;
    int board_y = (int)(point_vec.y) / 8;
    int bit_x = (int)(point_vec.x) % 8;
    int bit_y = 7 - (int)(point_vec.y) % 8;

    /*
    float decimal_x = point_vec.x - floor(point_vec.x);
    float decimal_y = point_vec.y - floor(point_vec.y);
    if(decimal_x > 0.8 && bit_x == 7) {
        pixels[pixel_idx] = 0xffff0000;
        return;
    }
    if(decimal_y > 0.8 && (int)(point_vec.y) % 8 == 7) {
        pixels[pixel_idx] = 0xffff0000;
        return;
    }
    */

    int board_idx = board_y * w_bitboards + board_x;
    Bitboard board_cell = board[board_idx];
    Bitboard mask = (Bitboard)1 << (bit_y * 8 + bit_x);
    if (board_cell & mask) {
        pixels[pixel_idx] = 0xFFFFFFFF; // White
    } else {
        pixels[pixel_idx] = 0xFF000000; // Black
    }
}

extern "C" void draw_conway(Bitboard* d_board, int w_bitboards, int h_bitboards, unsigned int* h_pixels, int pixels_w, int pixels_h, glm::vec2 lx_ty, glm::vec2 rx_by)
{
    size_t board_sz = w_bitboards * h_bitboards * sizeof(Bitboard);
    size_t pix_sz = pixels_w * pixels_h * sizeof(unsigned int);

    unsigned int* d_pixels;

    cudaMalloc((void**)&d_pixels, pix_sz);

    dim3 blockSize(16, 16);
    dim3 numBlocks((pixels_w + blockSize.x - 1) / blockSize.x, (pixels_h + blockSize.y - 1) / blockSize.y);
    conway_draw_kernel<<<numBlocks, blockSize>>>(d_board, w_bitboards, h_bitboards, d_pixels, pixels_w, pixels_h, lx_ty, rx_by);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
}

extern "C" void allocate_conway_grid(Bitboard** d_board, Bitboard** d_board_2, int w_bitboards, int h_bitboards) {
    size_t board_sz = w_bitboards * h_bitboards * sizeof(Bitboard);
    cudaMalloc((void**)d_board, board_sz);
    cudaMalloc((void**)d_board_2, board_sz);
    // Initialize with random data
    Bitboard* h_board = (Bitboard*)malloc(board_sz);
    for (int i = 0; i < w_bitboards * h_bitboards; i+=3) {
        h_board[i] = rand();
    }
    cudaMemcpy(*d_board, h_board, board_sz, cudaMemcpyHostToDevice);
}

extern "C" void free_conway_grid(Bitboard* d_board, Bitboard* d_board_2) {
    cudaFree(d_board);
    cudaFree(d_board_2);
}
