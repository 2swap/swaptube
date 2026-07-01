

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"

__global__ void two_render_real_valued_function(
    uint32_t* pixels, const Cuda::ivec2 wh,
    Cuda::ResolvedStateEquationComponent* xd_eq, int x_eq_size, float x_adjustment,
    Cuda::ResolvedStateEquationComponent* yd_eq, int y_eq_size, float y_adjustment,
    float dragger_x, float dragger_y,
    const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by
) {
    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;


    float distAccum = 0.0;
    
    // for(int dx = -1; dx <= 1; dx++) {
    //     for(int dy = -1; dy <= 1; dy++) {
    //     }
    // }
    //+ Cuda::vec2(dx*0.5, dy*0.5) 

    Cuda::vec2 point = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);


    float dragger_lerp = 0.0;
    float x_dist = (point.x - dragger_x);
    float y_dist = (point.y - dragger_y);
    float dragger_dist = x_dist*x_dist + y_dist*y_dist;

    if (dragger_dist < 0.06){
        pixels[pixel.y * wh.x + pixel.x] = 0xff000000;
        return;
    } else if (dragger_dist < 0.11){
        pixels[pixel.y * wh.x + pixel.x] = Cuda::colorlerp(0xff000000, 0xffffffff, (dragger_dist-0.06)*20);
        return;
    } else if (dragger_dist < 0.15){
        dragger_lerp = (0.15-dragger_dist)*25;
    }


    float cuda_tags[2] = { point.x, point.y };
    int error = 0;
    float x_eval = evaluate_resolved_state_equation(x_eq_size, xd_eq, cuda_tags, 2, error);
    float y_eval = evaluate_resolved_state_equation(y_eq_size, yd_eq, cuda_tags, 2, error);

    float minDist = 0.5;
    x_dist = abs(x_eval - round(x_eval));
    y_dist = abs(y_eval - round(y_eval));

    minDist = min(minDist, x_dist);
    minDist = min(minDist, y_dist);
    minDist = min(minDist, (y_dist*y_dist + x_dist*x_dist)*2.5);
    distAccum += minDist;


    float whiteness = max(0.0, 1.0f - distAccum*16.0);
    uint32_t color = Cuda::colorlerp(0xff000000, Cuda::OKLABtoRGB(255,1,x_eval*0.2/x_adjustment,y_eval*0.2/y_adjustment), whiteness);
    // uint32_t color = Cuda::OKLABtoRGB(255,whiteness,x_eval*0.2,y_eval*0.2);
    pixels[pixel.y * wh.x + pixel.x] =  Cuda::colorlerp(color,0xffffffff,dragger_lerp);
    // pixels[pixel.y * wh.x + pixel.x] = 0xff000000 | (color << 16) | (color << 8) | color;
}

extern "C" void two_d_algebra(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    Cuda::ResolvedStateEquationComponent* x_eq, int x_eq_size, float x_adjustment,
    Cuda::ResolvedStateEquationComponent* y_eq, int y_eq_size, float y_adjustment,
    float dragger_x, float dragger_y,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by
) {

    Cuda::ResolvedStateEquationComponent* xd_eq;
    size_t x_memsize = x_eq_size * sizeof(Cuda::ResolvedStateEquationComponent);
    cudaMalloc(&xd_eq, x_memsize);
    cudaMemcpy(xd_eq, x_eq, x_memsize, cudaMemcpyHostToDevice);

    Cuda::ResolvedStateEquationComponent* yd_eq;
    size_t y_memsize = y_eq_size * sizeof(Cuda::ResolvedStateEquationComponent);
    cudaMalloc(&yd_eq, y_memsize);
    cudaMemcpy(yd_eq, y_eq, y_memsize, cudaMemcpyHostToDevice);


    dim3 block_size(16, 16);
    dim3 grid_size((wh.x + block_size.x - 1) / block_size.x, (wh.y + block_size.y - 1) / block_size.y);
    two_render_real_valued_function<<<grid_size, block_size>>>( d_pixels, wh, 
        xd_eq, x_eq_size, x_adjustment,
        yd_eq, y_eq_size, y_adjustment,
        dragger_x, dragger_y,
        lx_ty, rx_by );

    cudaFree(xd_eq);
    cudaFree(yd_eq);
}


