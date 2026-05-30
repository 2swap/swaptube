#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"

__global__ void render_real_valued_function(
    uint32_t* pixels, const Cuda::ivec2 wh,
    Cuda::ResolvedStateEquationComponent* d_eq, int eq_size,
    const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by
) {
    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;

    int negative_evals = 0;
    int positive_evals = 0;

    for(int dx = -4; dx <= 4; dx++) {
        for(int dy = -4; dy <= 4; dy++) {
             Cuda::vec2 point = pixel_to_point_in_screen(pixel + Cuda::ivec2(dx, dy), lx_ty, rx_by, wh);
             float cuda_tags[1] = { point.x };
             int error = 0;
             float eval = evaluate_resolved_state_equation(eq_size, d_eq, cuda_tags, 1, error) - point.y;
             if (eval < 0) negative_evals++;
             if (eval > 0) positive_evals++;
        }
    }

    // Ensure positive_evals is the larger one
    if(negative_evals > positive_evals) {
        int temp = negative_evals;
        negative_evals = positive_evals;
        positive_evals = temp;
    }

    // The closer to equal, the whiter
    float whiteness = 1.0f - (positive_evals - negative_evals) / (float)(positive_evals + negative_evals);

    uint32_t color = d_colorlerp(0xff000000, 0xffffffff, whiteness );
    pixels[pixel.y * wh.x + pixel.x] = 0xFF000000 | (color << 16) | (color << 8) | color;
}

extern "C" void cuda_render_real_valued_function(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    Cuda::ResolvedStateEquationComponent* eq, int eq_size,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by
) {
    Cuda::ResolvedStateEquationComponent* d_eq;
    size_t memsize = eq_size * sizeof(Cuda::ResolvedStateEquationComponent);
    cudaMalloc(&d_eq, memsize);
    cudaMemcpy(d_eq, eq, memsize, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((wh.x + block_size.x - 1) / block_size.x, (wh.y + block_size.y - 1) / block_size.y);
    render_real_valued_function<<<grid_size, block_size>>>( d_pixels, wh, d_eq, eq_size, lx_ty, rx_by );

    cudaFree(d_eq);
}
