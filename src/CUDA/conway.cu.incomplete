#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <utility>

typedef uint64_t Bitboard;

Bitboard DRB = 0b00000000'01111111'01111111'01111111'01111111'01111111'01111111'01111111;
Bitboard DLB = 0b00000000'11111110'11111110'11111110'11111110'11111110'11111110'11111110;
Bitboard URB = 0b01111111'01111111'01111111'01111111'01111111'01111111'01111111'00000000;
Bitboard ULB = 0b11111110'11111110'11111110'11111110'11111110'11111110'11111110'00000000;
Bitboard DRC = 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'00000001;
Bitboard DLC = 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'10000000;
Bitboard URC = 0b00000001'00000000'00000000'00000000'00000000'00000000'00000000'00000000;
Bitboard ULC = 0b10000000'00000000'00000000'00000000'00000000'00000000'00000000'00000000;
Bitboard DRZ = 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'01111111;
Bitboard DLZ = 0b00000000'10000000'10000000'10000000'10000000'10000000'10000000'10000000;
Bitboard URZ = 0b00000001'00000001'00000001'00000001'00000001'00000001'00000001'00000000;
Bitboard ULZ = 0b11111110'00000000'00000000'00000000'00000000'00000000'00000000'00000000;
Bitboard DRS = 0b00000000'00000001'00000001'00000001'00000001'00000001'00000001'00000001;
Bitboard DLS = 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'11111110;
Bitboard URS = 0b01111111'00000000'00000000'00000000'00000000'00000000'00000000'00000000;
Bitboard ULS = 0b10000000'10000000'10000000'10000000'10000000'10000000'10000000'00000000;
Bitboard RW1 = 0b00000001'00000001'00000001'00000001'00000001'00000001'00000001'00000001;
Bitboard DW1 = 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'11111111;
Bitboard UW1 = 0b11111111'00000000'00000000'00000000'00000000'00000000'00000000'00000000;
Bitboard LW1 = 0b10000000'10000000'10000000'10000000'10000000'10000000'10000000'10000000;

Bitboard RW7 = ~LW1;
Bitboard DW7 = ~UW1;
Bitboard UW7 = ~DW1;
Bitboard LW7 = ~RW1;

__global__ void conway_kernel(unsigned int* board, int w_bitboards, int h_bitboards)
{
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

    Bitboard sul = (ul << 63 & ULC) | (cc >> 9 & DRB) | (uc << 55 & URS) | (cl >> 1 & DLZ);
    Bitboard suc = (uc << 56 & UW1) | (cc >> 8 & DW7);
    Bitboard sur = (ur << 49 & URC) | (cc >> 7 & DLB) | (uc << 57 & ULZ) | (cr >> 15 & DRS);
    Bitboard scl = (cl << 07 & LW1) | (cc >> 1 & RW7);
    Bitboard scc = cc;                                  
    Bitboard scr = (cr >> 07 & RW1) | (cc << 1 & LW7);
    Bitboard sdl = (dl >> 49 & DLC) | (cc << 7 & URB) | (dc >> 57 & DRZ) | (cl << 15 & ULS);
    Bitboard sdc = (dc >> 56 & DW1) | (cc << 8 & UW7);
    Bitboard sdr = (dr >> 63 & DRC) | (cc << 9 & ULB) | (dc >> 55 & DLS) | (cr << 1 & URZ);

    Bitboard half_add1_sum = sul ^ suc;
    Bitboard half_add1_car = sul & suc;
    Bitboard half_add2_sum = sur ^ scl;
    Bitboard half_add2_car = sur & scl;
    Bitboard half_add3_sum = scr ^ sdl;
    Bitboard half_add3_car = scr & sdl;
    Bitboard half_add4_sum = sdc ^ sdr;
    Bitboard half_add4_car = sdc & sdr;
}

extern "C" void iterate_conway(Bitboard* h_board, int w_bitboards, int h_bitboards)
{
    size_t board_sz = w_bitboards * h_bitboards * sizeof(Bitboard);

    cudaMalloc((void**)&d_pixels, pix_sz);
    cudaMemcpy(d_pixels, h_pixels, pix_sz, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_lines, ln_sz);
    cudaMemcpy(d_lines, h_lines, ln_sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_lines + blockSize - 1) / blockSize;
    render_lines_kernel<<<numBlocks, blockSize>>>(
        d_pixels, width, height,
        geom_mean_size, thickness, lines_opacity,
        d_lines, num_lines,
        camera_direction, camera_pos, conjugate_camera_direction, fov);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    cudaFree(d_lines);
}
