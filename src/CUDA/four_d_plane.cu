
#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"
#include "four_d_shared.cuh"

__device__ unsigned int four_d_color(Cuda::vec4 a, float brightness, Cuda::vec3 channels) {

    float ax = min(1.0,smallishness(a.y,brightness))*channels.x;
    float ay = min(1.0,smallishness(a.z,brightness))*channels.y;
    float az = min(1.0,smallishness(a.w,brightness))*channels.z;

    return Cuda::OKLABtoRGB(
        min(1.0,(ax+ay+az)*0.5)*255,
        1.0,
        (ax-ay)*0.866,
        (ax+ay)*0.5-az
    );
}

__device__ Cuda::vec4 four_d_plane_function(Cuda::vec4 v, const int equation, float commute, Cuda::vec4 jj, Cuda::vec4 ijj) {

    if (equation == 1){

        Cuda::vec4 sinv = v;
        Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
        Cuda::vec4 v_pow = v;

        for (int t = 3; t < 60; t+=2){
            v_pow = four_d_mult(v_pow, v2,commute,jj,ijj)/((1.0-t)*t);
            sinv += v_pow;  
            if (abs(v_pow.x) > 100000000){
                return sinv;
            }
        }
        // if (to_print){
        //     printf("%f %f %f %f\n",sinv.x,sinv.y,sinv.z,sinv.w);
        // }
        return sinv;

    } else if (equation == 2){

        Cuda::vec4 cosv(1,0,0,0);
        Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
        Cuda::vec4 v_pow = v;

        for (int t = 2; t < 60; t+=2){
            v_pow = four_d_mult(v_pow, v2,commute,jj,ijj)/((1.0-t)*t);
            cosv = cosv + v_pow;
            if (abs(v_pow.x) > 100000000){
                return cosv;//Cuda::vec4(100000000,100000000,100000000,100000000) ;
            }
        }
        return cosv;

    }

    Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
    Cuda::vec4 v3 = four_d_mult(v,v2,commute,jj,ijj);
    Cuda::vec4 v4 = four_d_mult(v2,v2,commute,jj,ijj);

    Cuda::vec4 v5 = four_d_mult(v3,v2,commute,jj,ijj);
    Cuda::vec4 v6 = four_d_mult(v4,v2,commute,jj,ijj);
    Cuda::vec4 v7 = four_d_mult(v5,v2,commute,jj,ijj);
    Cuda::vec4 v8 = four_d_mult(v6,v2,commute,jj,ijj);
    Cuda::vec4 v9 = four_d_mult(v7,v2,commute,jj,ijj);
    Cuda::vec4 v10 = four_d_mult(v5,v5,commute,jj,ijj);
    Cuda::vec4 v12 = four_d_mult(v7,v5,commute,jj,ijj);



    if (equation == 0){
        return 1 + v + v2/2 + v3/6 + v4/24 + v5/120 + v6/720 + v7/5040 + v8/40320;

    } else if (equation == 3){
        // return v - v2 - v5 + v10;
        return v2 - v4 - v6 + v12;
        // return -2 + v*6 - v2*2 - v3*3 + v6;
        // return 1 + v + v2 + v3 + v4;
        // return 1 - v + v3 - v4 + v5 - v7 + v8;
        // return 1 - v + v2 - v3 + v4;

    }
    
    return v;

}



__global__ void four_d_plane_kernel(
    const Cuda::ivec2 wh,
    const Cuda::vec2 lx_ty,
    const Cuda::vec2 rx_by,
    const Cuda::vec4 x_unit,
    const Cuda::vec4 y_unit,

    const Cuda::vec4 jj,
    const Cuda::vec4 ijj,

    const float brightness,
    const Cuda::vec3 channels,

    unsigned int internal_color,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);
    Cuda::vec2 point_vec = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);
    
    Cuda::vec4 four_d_ouput = four_d_plane_function((point_vec.x*x_unit+point_vec.y*y_unit),2,1.0,jj,ijj); 

    colors[pixel_y * wh.x + pixel_x] = four_d_color(four_d_ouput,brightness,channels); 

}

// Host function to launch the kernel
extern "C" void four_d_plane_render(
    const Cuda::ivec2& wh,
    const Cuda::vec2& lx_ty,
    const Cuda::vec2& rx_by,
    Cuda::vec4 x_unit,
    Cuda::vec4 y_unit,

    const Cuda::vec4 jj,
    const Cuda::vec4 ijj,

    const float brightness,
    const Cuda::vec3 channels,

    unsigned int internal_color,
    unsigned int* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    four_d_plane_kernel<<<numBlocks, threadsPerBlock>>>(
        wh, lx_ty, rx_by,
        x_unit, y_unit,
        jj,ijj,
        brightness, channels,
        internal_color, d_colors
    );
    cudaDeviceSynchronize();
}
