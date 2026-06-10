#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <utility>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/Color.h"

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

__global__ void conway_kernel(Bitboard* board, Bitboard* board_2, const Cuda::ivec2 grid_wh_bitboards) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= grid_wh_bitboards.x * grid_wh_bitboards.y) return;

    int bx = idx % grid_wh_bitboards.x;
    int by = idx / grid_wh_bitboards.x;
    int d = (by + 1) % grid_wh_bitboards.y;
    int l = (bx + 1) % grid_wh_bitboards.x;
    int u = (by + grid_wh_bitboards.y - 1) % grid_wh_bitboards.y;
    int r = (bx + grid_wh_bitboards.x - 1) % grid_wh_bitboards.x;

    auto load = [&](int y, int x) -> Bitboard {
        if (x < 0 || x >= grid_wh_bitboards.x ||
            y < 0 || y >= grid_wh_bitboards.y)
            return 0;
        return board[y * grid_wh_bitboards.x + x];
    };

    Bitboard ul = load(u, l);
    Bitboard uc = load(u, bx);
    Bitboard ur = load(u, r);

    Bitboard cl = load(by, l);
    Bitboard cc = board[idx];
    Bitboard cr = load(by, r);

    Bitboard dl = load(d, l);
    Bitboard dc = load(d, bx);
    Bitboard dr = load(d, r);

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

extern "C" void iterate_conway(Bitboard*& d_board, Bitboard*& d_board_2, const Cuda::ivec2& grid_wh_bitboards, const int iterations)
{
    dim3 blockSize(256);
    dim3 numBlocks((grid_wh_bitboards.x * grid_wh_bitboards.y + blockSize.x - 1) / blockSize.x);
    for(int i = 0; i < iterations; i++) {
        conway_kernel<<<numBlocks, blockSize>>>(d_board, d_board_2, grid_wh_bitboards);
        cudaDeviceSynchronize();
        Bitboard* temp = d_board;
        d_board = d_board_2;
        d_board_2 = temp;
    }
}

__device__ void count_neighbors_8(
    Bitboard b,
    Bitboard& ones,
    Bitboard& twos,
    Bitboard& fours,
    Bitboard& eights)
{
    Bitboard left  = (b << 1) | (b >> 7);
    Bitboard right = (b >> 1) | (b << 7);
    Bitboard up    = (b << 8) | (b >> 56);
    Bitboard down  = (b >> 8) | (b << 56);

    Bitboard ul = (b << 9)  | (b >> 55);
    Bitboard ur = (b << 7)  | (b >> 57);
    Bitboard dl = (b >> 7)  | (b << 57);
    Bitboard dr = (b >> 9)  | (b << 55);

    // This is your exact “8-direction sum pipeline”
    Bitboard sul = ul;
    Bitboard suc = up;
    Bitboard sur = ur;
    Bitboard scl = left;
    Bitboard scr = right;
    Bitboard sdl = dl;
    Bitboard sdc = down;
    Bitboard sdr = dr;

    // half-adder layer (same as your kernel style)
    Bitboard o1 = sul ^ suc;
    Bitboard c1 = sul & suc;

    Bitboard o2 = sur ^ scl;
    Bitboard c2 = sur & scl;

    Bitboard o3 = sdl ^ sdc;
    Bitboard c3 = sdl & sdc;

    Bitboard o4 = sdr ^ scr;
    Bitboard c4 = sdr & scr;

    // combine pairs
    Bitboard o12 = o1 ^ o2;
    Bitboard c12 = (o1 & o2) | (c1 | c2);

    Bitboard o34 = o3 ^ o4;
    Bitboard c34 = (o3 & o4) | (c3 | c4);

    Bitboard ones_part = o12 ^ o34;
    Bitboard carry1 = (o12 & o34) | (c12 | c34);

    Bitboard twos_part = carry1;   // matches your encoding
    Bitboard fours_part = 0;       // (depends on full cascade in your kernel)
    Bitboard eights_part = 0;

    ones = ones_part;
    twos = twos_part;
    fours = fours_part;
    eights = eights_part;
}

__device__ Bitboard conway_rule(
    Bitboard alive,
    Bitboard ones,
    Bitboard twos,
    Bitboard fours,
    Bitboard eights)
{
    Bitboard is_two =
        (~ones) & twos & ~fours & ~eights;

    Bitboard is_three =
        ones & twos & ~fours & ~eights;

    return (alive & is_two) | is_three;
}

__device__ Bitboard step8x8(Bitboard b)
{
    Bitboard ones, twos, fours, eights;
    count_neighbors_8(b, ones, twos, fours, eights);

    return conway_rule(b, ones, twos, fours, eights);
}

// Given a single bitboard and its target,
__global__ void reverse_conway_kernel(
    Bitboard board,
    Bitboard target,
    Bitboard* best_board,
    int* best_score)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (1 << 24)) return; // 64^4 combos

    int a = (idx >> 18) & 0x3f;
    int b = (idx >> 12) & 0x3f;
    int c = (idx >> 6)  & 0x3f;
    int d = (idx)       & 0x3f;

    Bitboard flip =
        ((Bitboard)1 << a) |
        ((Bitboard)1 << b) |
        ((Bitboard)1 << c) |
        ((Bitboard)1 << d);

    Bitboard flipped = board ^ flip;

    // run forward CGOL (greedy single-step model)
    Bitboard result = step8x8(flipped);

    int score = __popcll(result ^ target);

    // atomic best update
    atomicMin(best_score, score);

    __syncthreads();

    if (*best_score == score)
    {
        *best_board = flip;
    }
}

// Kernel to find the first bit which differs on two large conway grids.
__global__ void find_first_different_bit(
    Bitboard* board,
    Bitboard* target,
    int* first_different_bit,
    const Cuda::ivec2 grid_wh_bitboards)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int totalBits = grid_wh_bitboards.x * grid_wh_bitboards.y * 64;
    if (idx >= totalBits) return;

    int bitboard_idx = idx / 64;
    int bit_idx = idx % 64;

    Bitboard b = board[bitboard_idx];
    Bitboard t = target[bitboard_idx];

    int diff = ((b >> bit_idx) & 1ULL) != ((t >> bit_idx) & 1ULL);

    if (diff)
    {
        atomicMin(first_different_bit, idx);
    }
}

// Run CGOL in reverse
// Find the first incorrectly predicted bit in the output.
// Isolate the 8x8 grid (one bitboard) around that bit
// in the input and the output.
// Generate all of the 64^4 possible quadruplets of bits in the bitboard,
// flip them in the board, and see how close the result is to the target.
// We measure success as popcnt(result XOR target).
// Choose the one that minimizes this value, and flip those 4 bits.
extern "C" void reverse_conway_loop(
    Bitboard*& d_board,
    Bitboard*& d_board_2,
    Bitboard*& target,
    const Cuda::ivec2& grid_wh_bitboards,
    int max_iters)
{
    dim3 blockSize(256);
    dim3 numBlocks(
        (grid_wh_bitboards.x * grid_wh_bitboards.y + blockSize.x - 1) / blockSize.x
    );

    for (int iter = 0; iter < max_iters; iter++)
    {
        // 1. forward step
        conway_kernel<<<numBlocks, blockSize>>>(
            d_board, d_board_2, grid_wh_bitboards);
        cudaDeviceSynchronize();

        std::swap(d_board, d_board_2);

        // 2. find first mismatch
        int* d_first;
        cudaMalloc(&d_first, sizeof(int));

        int init = INT_MAX;
        cudaMemcpy(d_first, &init, sizeof(int), cudaMemcpyHostToDevice);

        find_first_different_bit<<<numBlocks, blockSize>>>(
            d_board, target, d_first, grid_wh_bitboards);

        cudaDeviceSynchronize();

        int h_first;
        cudaMemcpy(&h_first, d_first, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_first);

        // 3. stop condition
        if (h_first == INT_MAX)
        {
            break; // fully matched
        }

        // 4. map bit index → bitboard index
        int board_idx = h_first / 64;

        Bitboard host_board, host_target;

        cudaMemcpy(&host_board,
                   &d_board[board_idx],
                   sizeof(Bitboard),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(&host_target,
                   &target[board_idx],
                   sizeof(Bitboard),
                   cudaMemcpyDeviceToHost);

        // 5. run local reverse search (your kernel)
        Bitboard* d_best;
        int* d_best_score;

        cudaMalloc(&d_best, sizeof(Bitboard));
        cudaMalloc(&d_best_score, sizeof(int));

        int INF = 1e9;
        cudaMemcpy(d_best_score, &INF, sizeof(int), cudaMemcpyHostToDevice);

        dim3 block2(256);
        dim3 grid2((1 << 24) / 256);

        reverse_conway_kernel<<<grid2, block2>>>(
            host_board,
            host_target,
            d_best,
            d_best_score);

        cudaDeviceSynchronize();

        Bitboard best_flip;
        cudaMemcpy(&best_flip, d_best, sizeof(Bitboard), cudaMemcpyDeviceToHost);

        cudaFree(d_best);
        cudaFree(d_best_score);

        // 6. apply fix
        d_board[board_idx] ^= best_flip;
    }
}

__global__ void conway_draw_kernel_old(Bitboard* board, Bitboard* board_2, const Cuda::ivec2 grid_wh_bitboards, unsigned int* pixels, const Cuda::ivec2 pix_wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, float w_t)
{
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px >= pix_wh.x || py >= pix_wh.y) return;

    int pixel_idx = py * pix_wh.x + px;
    Cuda::vec2 point_vec = pixel_to_point_in_screen(Cuda::vec2(px, py), lx_ty, rx_by, pix_wh);
    point_vec += 4*grid_wh_bitboards; // Center

    if(point_vec.x < 0 || point_vec.y < 0) {
        pixels[pixel_idx] = 0xff808080;
        return;
    }
    if(point_vec.x >= grid_wh_bitboards.x * 8 || point_vec.y >= grid_wh_bitboards.y * 8) {
        pixels[pixel_idx] = 0xff808080;
        return;
    }

    int mboard_x =     (int)(point_vec.x - 1) / 8;
    int mboard_y =     (int)(point_vec.y - 1) / 8;
    int   mbit_x =     (int)(point_vec.x - 1) % 8;
    int   mbit_y = 7 - (int)(point_vec.y - 1) % 8;

    int board_x =     (int)(point_vec.x) / 8;
    int board_y =     (int)(point_vec.y) / 8;
    int   bit_x =     (int)(point_vec.x) % 8;
    int   bit_y = 7 - (int)(point_vec.y) % 8;

    int pboard_x =     (int)(point_vec.x + 1) / 8;
    int pboard_y =     (int)(point_vec.y + 1) / 8;
    int   pbit_x =     (int)(point_vec.x + 1) % 8;
    int   pbit_y = 7 - (int)(point_vec.y + 1) % 8;

    /*
    // Highlight bit boundaries
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


    // Trilinear Interpolation
    float w_x = point_vec.x - floor(point_vec.x);
    float w_y = point_vec.y - floor(point_vec.y);

    bool b000 = board[board_y * grid_wh_bitboards.x + board_x] & ((Bitboard)1 << (bit_y * 8 + bit_x));
    bool b100 = board_2[board_y * grid_wh_bitboards.x + board_x] & ((Bitboard)1 << (bit_y * 8 + bit_x));

    if(b000 == b100) {
        if(b000) {
            pixels[pixel_idx] = 0xFFFFFFFF; // White
        } else {
            pixels[pixel_idx] = 0xFF000000; // Black
        }
        return;
    }

    bool tpx = board[board_y * grid_wh_bitboards.x + pboard_x] & ((Bitboard)1 << (bit_y * 8 + pbit_x));
    bool tpy = board[pboard_y * grid_wh_bitboards.x + board_x] & ((Bitboard)1 << (pbit_y * 8 + bit_x));
    bool tmx = board[board_y * grid_wh_bitboards.x + mboard_x] & ((Bitboard)1 << (bit_y * 8 + mbit_x));
    bool tmy = board[mboard_y * grid_wh_bitboards.x + board_x] & ((Bitboard)1 << (mbit_y * 8 + bit_x));

    float dpx = .1 + 1 - w_x;
    float dmx = .1 + w_x;
    float dpy = .1 + 1 - w_y;
    float dmy = .1 + w_y;
    float dpt = .1 + 1 - w_t;
    float dmt = .1 + w_t;

    float amount_vote_on = 0.0f;
    float amount_vote_off = 0.0f;
    if(tpx) amount_vote_on += 1/dpx; else amount_vote_off += 1/dpx;
    if(tmx) amount_vote_on += 1/dmx; else amount_vote_off += 1/dmx;
    if(tpy) amount_vote_on += 1/dpy; else amount_vote_off += 1/dpy;
    if(tmy) amount_vote_on += 1/dmy; else amount_vote_off += 1/dmy;
    if(b100) amount_vote_on += 2/dpt; else amount_vote_off += 2/dpt;
    if(b000) amount_vote_on += 2/dmt; else amount_vote_off += 2/dmt;

    float proportion = (amount_vote_on / (amount_vote_on + amount_vote_off));

    float aa = Cuda::lerp(b000, proportion, w_t);
    float bb = Cuda::lerp(proportion, b100, w_t);
    float f = Cuda::lerp(aa, bb, w_t);

    if (f > .5) {
        pixels[pixel_idx] = 0xFFFFFFFF; // White
    } else {
        pixels[pixel_idx] = 0xFF000000; // Black
    }
}

__device__ Bitboard rect_mask(
    int x0,
    int x1,
    int y0,
    int y1)
{
    Bitboard mask = 0;

    for(int y = y0; y <= y1; y++)
    {
        uint8_t row =
            ((1u << (x1 - x0 + 1)) - 1u)
            << x0;

        int bit_y = 7 - y;

        mask |= ((Bitboard)row)
              << (bit_y * 8);
    }

    return mask;
}

__global__ void conway_draw_kernel(Bitboard* board, Bitboard* board_2, const Cuda::ivec2 grid_wh_bitboards, unsigned int* pixels, const Cuda::ivec2 pix_wh, Cuda::vec2 lx_ty, Cuda::vec2 rx_by, float w_t)
{
    Cuda::ivec2 pixel(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    if (pixel.x >= pix_wh.x || pixel.y >= pix_wh.y) return;

    // This version, we assume that the board is so zoomed out that
    // each pixel hovers over more than one conway cell. We compute
    // all of the cells that the pixel hovers over, and do a weighted
    // average of their states to determine the pixel color. This is a form
    // of anti-aliasing.
    // To count the votes, we use the popcnt instruction on masked sections
    // of bitboards underlying this pixel.
    //
    // First, compute the coordinates which this pixel intersects with.

    Cuda::vec2 p00 = pixel_to_point_in_screen(pixel + Cuda::ivec2(0, 0), lx_ty, rx_by, pix_wh);
    Cuda::vec2 p11 = pixel_to_point_in_screen(pixel + Cuda::ivec2(1, 1), lx_ty, rx_by, pix_wh);

    p00 += 4 * grid_wh_bitboards;
    p11 += 4 * grid_wh_bitboards;

    int min_x = min(p00.x, p11.x);
    int max_x = max(p00.x, p11.x);
    int min_y = min(p00.y, p11.y);
    int max_y = max(p00.y, p11.y);

    int min_x_bitboard = max(min_x / 8, 0);
    int max_x_bitboard = min(max_x / 8, grid_wh_bitboards.x - 1);
    int min_y_bitboard = max(min_y / 8, 0);
    int max_y_bitboard = min(max_y / 8, grid_wh_bitboards.y - 1);

    int min_x_bit = min_x % 8;
    int min_y_bit = min_y % 8;
    int max_x_bit = max_x % 8;
    int max_y_bit = max_y % 8;

    int alive = 0;
    int total = 0;

    for(int by = min_y_bitboard; by <= max_y_bitboard; by++) {
        for(int bx = min_x_bitboard; bx <= max_x_bitboard; bx++) {
            Bitboard b = board[by * grid_wh_bitboards.x + bx];
            int lx0 = 0;
            int lx1 = 7;
            int ly0 = 0;
            int ly1 = 7;

            if(bx == min_x_bitboard)
                lx0 = min_x_bit;

            if(bx == max_x_bitboard)
                lx1 = max_x_bit;

            if(by == min_y_bitboard)
                ly0 = min_y_bit;

            if(by == max_y_bitboard)
                ly1 = max_y_bit;

            Bitboard mask =
                rect_mask(
                    lx0,
                    lx1,
                    ly0,
                    ly1);

            alive += __popcll(b & mask);
            total += __popcll(mask);
        }
    }

    int pixel_idx = pixel.y * pix_wh.x + pixel.x;

    if(total == 0) {
        pixels[pixel_idx] = 0xff808080;
        return;
    }

    float ratio = (float)alive / total;
    ratio = (1 - ratio);
    ratio *= ratio;
    ratio *= ratio;
    ratio = 1 - ratio;
    pixels[pixel_idx] = Cuda::colorlerp(0xFF000000, 0xFFFFFFFF, ratio);
}

extern "C" void draw_conway(Bitboard* d_board, Bitboard* d_board_2, const Cuda::ivec2& grid_wh_bitboards, uint32_t* d_pixels, const Cuda::ivec2& pix_wh, const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by, float transition)
{
    dim3 blockSize(16, 16);
    dim3 numBlocks((pix_wh.x + blockSize.x - 1) / blockSize.x, (pix_wh.y + blockSize.y - 1) / blockSize.y);
    conway_draw_kernel<<<numBlocks, blockSize>>>(d_board, d_board_2, grid_wh_bitboards, d_pixels, pix_wh, lx_ty, rx_by, transition);
    cudaDeviceSynchronize();
}

__global__ void initialize_boards(Bitboard* d_board, Bitboard* d_board_2, const Cuda::ivec2 grid_wh_bitboards) {
    Cuda::ivec2 idx(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);

    Bitboard board_value = ((uint64_t)idx.y) << 32 | idx.x;

    int index = idx.y * grid_wh_bitboards.x + idx.x;
    d_board[index] = d_board_2[index] = board_value;
}

extern "C" void allocate_conway_grid(Bitboard** d_board, Bitboard** d_board_2, const Cuda::ivec2& grid_wh_bitboards) {
    size_t board_sz = grid_wh_bitboards.x * grid_wh_bitboards.y * sizeof(Bitboard);
    cudaMalloc((void**)d_board, board_sz);
    cudaMalloc((void**)d_board_2, board_sz);

    // Use kernel to spawn initial board state from envelope texture.
    dim3 blockSize(16, 16);
    dim3 numBlocks((grid_wh_bitboards.x + blockSize.x - 1) / blockSize.x, (grid_wh_bitboards.y + blockSize.y - 1) / blockSize.y);
    initialize_boards<<<numBlocks, blockSize>>>(*d_board, *d_board_2, grid_wh_bitboards);
}

extern "C" void free_conway_grid(Bitboard* d_board, Bitboard* d_board_2) {
    cudaFree(d_board);
    cudaFree(d_board_2);
}
