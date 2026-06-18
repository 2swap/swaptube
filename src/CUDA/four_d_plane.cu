/*
#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
// #include "four_d_algebra.cu"

__device__ unsigned int four_d_color(Cuda::vec4 a, float brightness) {

    // float real_smallness = 1;//0.6+0.4*smallness(a.x,brightness);
    float real_smallness = 0.6+0.4*smallness(a.x,brightness);
    // real_smallness = real_smallness*real_smallness*real_smallness;

    return 255 << 24 |
    (((unsigned int)(real_smallness*smallness(a.y,brightness)*255) & 0xff) << 16) |
    (((unsigned int)(real_smallness*smallness(a.z,brightness)*255) & 0xff) << 8) |
    (((unsigned int)(real_smallness*smallness(a.w,brightness)*255) & 0xff) << 0);
}


__device__ Cuda::vec4 four_d_plane_function(Cuda::vec4 v) {

    //quaternion
    // return v;
    Cuda::vec4 v2 = four_d_mult(v,v);
    Cuda::vec4 v3 = four_d_mult(v,v2);
    Cuda::vec4 v4 = four_d_mult(v2,v2);
    Cuda::vec4 v5 = four_d_mult(v3,v2);
    Cuda::vec4 v6 = four_d_mult(v4,v2);
    Cuda::vec4 v7 = four_d_mult(v5,v2);
    Cuda::vec4 v8 = four_d_mult(v6,v2);
    Cuda::vec4 v9 = four_d_mult(v7,v2);

    // Cuda::vec4 v11 = four_d_mult(v9,v2);
    // Cuda::vec4 v15 = four_d_mult(v9,v6);

    Cuda::vec4 m1 = v4+four_d_real(3);
    m1 = four_d_mult( m1, v2-four_d_real(4));
    m1 = four_d_mult( m1, v3+four_d_real(3));
    return m1;


    // Cuda::vec4 sinv = v;
    // Cuda::vec4 v2 = four_d_mult(v,v);
    // Cuda::vec4 v_pow = v;

    // for (int t = 3; t < 30; t+=2){
    //     v_pow = four_d_mult(v_pow, v2)/(-1.0*(t-1.0)*t);
    //     sinv = sinv + v_pow;
    // }
    // return sinv;

    // - v3/6 + v5/120 - v7/5040 + v9/362880 - v11/39916800;   



    // Cuda::vec4 cosv(1,0,0,0);
    // Cuda::vec4 v2 = four_d_mult(v,v);
    // Cuda::vec4 v_pow = v;

    // for (int t = 2; t < 31; t+=2){
    //     v_pow = four_d_mult(v_pow, v2)/(-1.0*(t-1.0)*t);
    //     cosv = cosv + v_pow;
    // }
    // return cosv;
    // Cuda::vec4 cosv = 1 - v2/2.0 + v4/24.0 - v7/840.0 + v8/40320.0 - v10/3628800;



    // Cuda::vec4 coeff(0.2,1.0,0.2,-1.4);
    // Cuda::vec4 cv4 = four_d_mult(coeff,cv4);
    // return v7+4.0*v6-10*v5+20*v-16.0;
    // return v15-7.0*v5-5.0*v7+four_d_real(35.0);
    // return v6-v4+10.0*v3+1.0;
    // return v18*0.01 - v7 + v6*2.7 +  v5*8.0 - v3*60.0 -50.0;
    // return 1 + v + v2/2 + v3/6 + v4/24 + v5/120 + v6/720 + v7/5040 + v8/40320;
    // return sinv*0.8 + cosv*1.6;
    // return sinv;
}

__global__ void four_d_plane_kernel(
    const Cuda::ivec2 wh,
    const Cuda::vec2 lx_ty,
    const Cuda::vec2 rx_by,
    const Cuda::vec4 x_unit,
    const Cuda::vec4 y_unit,
    const float brightness,

    unsigned int internal_color,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);
    Cuda::vec2 point_vec = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);
    
    Cuda::vec4 four_d_ouput = four_d_plane_function((point_vec.x*x_unit+point_vec.y*y_unit)); 

    colors[pixel_y * wh.x + pixel_x] = four_d_color(four_d_ouput,brightness); 

}

// Host function to launch the kernel
extern "C" void four_d_plane_render(
    const Cuda::ivec2& wh,
    const Cuda::vec2& lx_ty,
    const Cuda::vec2& rx_by,
    Cuda::vec4 x_unit,
    Cuda::vec4 y_unit,
    const float brightness,
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
        brightness,
        internal_color, d_colors
    );
    cudaDeviceSynchronize();
}
*/