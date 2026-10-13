

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
// #include "four_d_shared.cuh"



__device__ __forceinline__ float smallness_2d(float s, float brightness){
    // return 1/(1+0.02*s*s*abs(s));
    float s_sq = s*s;
    return 1/(1+s_sq/brightness);
    // return 1/(1+s*s*s*s);
}



__device__ uint32_t two_d_to_color(Cuda::vec2 a, Cuda::vec2 channels,int brightness) {

    // float ax = min(1.0,fade*a.x);
    // float ay = min(1.0,fade*a.y);
    // float az = min(1.0,fade*a.z);

    float ax = min(1.0,smallness_2d(a.y,8.0))*channels.y;
    float ay = min(1.0,smallness_2d(a.x,8.0))*channels.x;
    float az = 0.1;

    return Cuda::OKLABtoRGB(
        min(1.0,(ax+ay+az)*0.5)*brightness,
        1.0,
        (ax-ay)*0.92,
        (ax+ay)*0.4-az
    );
    
}

__device__ Cuda::vec2 two_d_mult(Cuda::vec2 a, Cuda::vec2 b,  Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yy){
    return a.x*b.x*xx + (a.x*b.y + a.y*b.x)*xy + a.y*b.y*yy;
}

__device__ Cuda::vec2 two_d_function(Cuda::vec2 v, const int equation, const float equation_lerp, Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yy) {

    if (equation_lerp != 0){

        return two_d_function(v, equation, 0, xx, xy, yy)*(1-equation_lerp) +  two_d_function(v, equation+1, 0, xx, xy, yy)*equation_lerp;
    }

    if (equation == 3){

        Cuda::vec2 sinv = v;
        Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        Cuda::vec2 v_pow = v;

        for (int t = 3; t < 60; t+=2){
            v_pow = two_d_mult(v_pow, v2,xx,xy,yy)/((1.0-t)*t);
            sinv += v_pow;  
            if (abs(v_pow.x) > 100000000){
                return Cuda::vec2(100000000,100000000) ;
            }
        }
        // if (to_print){
        //     printf("%f %f %f %f\n",sinv.x,sinv.y,sinv.z,sinv.w);
        // }
        return sinv;

    } else if (equation == 2){

        // Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        // Cuda::vec2 v3 = two_d_mult(v,v2,xx,xy,yy);
        // Cuda::vec2 v5 = two_d_mult(v3,v2,xx,xy,yy);

        // return v2-3*v;

        Cuda::vec2 cosv(1,0);
        Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        Cuda::vec2 v_pow = v;

        for (int t = 2; t < 60; t+=2){
            v_pow = two_d_mult(v_pow, v2,xx,xy,yy)/((1.0-t)*t);
            cosv = cosv + v_pow;
        }
        return cosv;

    } else if (equation == 101){
        return v;
    } else if (equation == 102){
        return two_d_mult(v,v,xx,xy,yy);
    } else if (equation == 103){
        Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        return two_d_mult(v,v2,xx,xy,yy);
    } else if (equation == 104){
        Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        return two_d_mult(v2,v2,xx,xy,yy);
    }

    Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
    Cuda::vec2 v3 = two_d_mult(v,v2,xx,xy,yy);
    Cuda::vec2 v4 = two_d_mult(v2,v2,xx,xy,yy);

    Cuda::vec2 v5 = two_d_mult(v3,v2,xx,xy,yy);
    Cuda::vec2 v6 = two_d_mult(v4,v2,xx,xy,yy);
    Cuda::vec2 v7 = two_d_mult(v5,v2,xx,xy,yy);
    Cuda::vec2 v8 = two_d_mult(v6,v2,xx,xy,yy);
    Cuda::vec2 v9 = two_d_mult(v7,v2,xx,xy,yy);
    Cuda::vec2 v10 = two_d_mult(v5,v5,xx,xy,yy);
    Cuda::vec2 v12 = two_d_mult(v7,v5,xx,xy,yy);



    if (equation == 105){
        // return v - v2 - v5 + v10;
        // return v5/256 - 4;
        // return -2 + v*6 - v2*2 - v3*3 + v6;
        // return 1 + v + v2 + v3 + v4;
        // return 1 - v + v3 - v4 + v5 - v7 + v8;
        return v - v2 + v3 - v4 + v5;

    } else if (equation == 106){
        return v - v3/6 + v5/120 - v7/5040 + v9/362880;

    } else if (equation == 107){
        return Cuda::vec2(1,0) - v2/2 + v4/24 - v6/720 + v8/40320;


    } else if (equation == 4){
        // return v - v2 - v5 + v10;
        return v5/512 - 4;
        // return -2 + v*6 - v2*2 - v3*3 + v6;
        // return 1 + v + v2 + v3 + v4;
        // return 1 - v + v3 - v4 + v5 - v7 + v8;
        // return 1 - v + v2 - v3 + v4;

    } else if (equation == 5){
        // return v7/2048 - 3;
        return v9/1600000 - 4 - v3/32;

    } else if (equation == 6){
        // return two_d_function(v3*0.02, 3, 0, xx, xy, yy) - v2*0.04;
        // return two_d_function(v3*0.02, 3, 0, xx, xy, yy)*0.3 - 2;
        Cuda::vec2 vr = two_d_mult(v,Cuda::vec2(0.8,0.6),xx,xy,yy)+Cuda::vec2(2,2.6);
        return two_d_function(v, 3, 0, xx, xy, yy)*0.5 + two_d_function(vr, 3, 0, xx, xy, yy)*0.4;
        // Cuda::vec2 s = two_d_function(v4*0.0002, 3, 0, xx, xy, yy);
        // return two_d_mult(s,s,xx,xy,yy);

    }  else if (equation == 8){
        return two_d_function(v2*0.06, 2, 0, xx, xy, yy) + two_d_function(v*0.4, 3, 0, xx, xy, yy)*2;
        // return two_d_function(v2*0.05, 2, 0, xx, xy, yy) + two_d_function(v*0.4, 3, 0, xx, xy, yy)*2;
        // Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        // Cuda::vec2 v3 = two_d_mult(v,v2,xx,xy,yy);
        // Cuda::vec2 v2z = two_d_mult(Cuda::vec2(0.02,0.001),v3,xx,xy,yy);
        // return two_d_function(v2z, 2, 0, xx, xy, yy)*2;
    }   else if (equation == 7){

        Cuda::vec2 vr = two_d_mult(v,Cuda::vec2(-0.4,0.8),xx,xy,yy);
        return two_d_function(v2*0.1, 3, 0, xx, xy, yy)*0.5 + two_d_function(vr*0.6, 2, 0, xx, xy, yy)*0.4;
  
    } 
    
    
    
    return v;

}

__global__ void two_d_smallness_graph(
    uint32_t* pixels, const Cuda::ivec2 wh,
    int equation, float equation_lerp,
    Cuda::vec2 channels,
    Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yy, 
     int brightness,
    const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by
) {

    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;

    Cuda::vec2 point = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);

    Cuda::vec2 op_output = two_d_function(point*3, equation, equation_lerp, xx, xy, yy);

    pixels[pixel.y * wh.x + pixel.x] =  two_d_to_color(op_output,channels,brightness);

}


extern "C" void two_d_algebra(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    // Cuda::vec2 dragger_pos, 
    // float dragger_type, float dragger_brightness, 
    // Cuda::vec4 dragger_inverse,
    int equation, float equation_lerp,
    Cuda::vec2 channels, 
    Cuda::vec2 xx,  Cuda::vec2 xy,  Cuda::vec2 yy, 
    // float number_line,
    int brightness,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by
) {


    dim3 block_size(16, 16);
    dim3 grid_size((wh.x + block_size.x - 1) / block_size.x, (wh.y + block_size.y - 1) / block_size.y);

    two_d_smallness_graph<<<grid_size, block_size>>>( d_pixels, wh, 
        equation, equation_lerp,
        channels, xx, xy, yy, brightness,
        lx_ty, rx_by );


}


