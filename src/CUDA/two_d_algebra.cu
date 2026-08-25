

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
#include "four_d_shared.cuh"




__device__ Cuda::vec2 two_d_operation(
    Cuda::vec2 point, Cuda::vec2 dragger_pos, float dragger_type, Cuda::vec4 dragger_inverse,
    float number_line
    // Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yx, Cuda::vec2 yy
){

    if (dragger_type == 0){
        return point;

    } else if (dragger_type == 1){
        if (number_line == 1){
            return Cuda::vec2(point.x-dragger_pos.x, point.y);
        }
        return point - dragger_pos;

    } else {
        return Cuda::vec2(
            point.x*dragger_inverse.x+point.y*dragger_inverse.y,
            point.x*dragger_inverse.z+point.y*dragger_inverse.w
        );
    }

}

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

__device__ Cuda::vec2 two_d_function(Cuda::vec2 v, const int equation, Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yy) {


    if (equation == 1){

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

        Cuda::vec2 cosv(1,0);
        Cuda::vec2 v2 = two_d_mult(v,v,xx,xy,yy);
        Cuda::vec2 v_pow = v;

        for (int t = 2; t < 41; t+=2){
            v_pow = two_d_mult(v_pow, v2,xx,xy,yy)/((1.0-t)*t);
            cosv = cosv + v_pow;
        }
        return cosv;

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



    if (equation == 3){
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

__global__ void two_d_smallness_graph(
    uint32_t* pixels, const Cuda::ivec2 wh,
    int mode, Cuda::vec2 channels,
    Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yy, 
     int brightness,
    const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by
) {

    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;

    Cuda::vec2 point = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);

    Cuda::vec2 op_output = two_d_function(point*3, mode, xx, xy, yy);

    pixels[pixel.y * wh.x + pixel.x] =  two_d_to_color(op_output,channels,brightness);

}

__global__ void two_d_algebra_grid(
    uint32_t* pixels, const Cuda::ivec2 wh,
    Cuda::vec2 dragger_pos, 
    float dragger_type, float dragger_brightness, 
    // float algebra,
    Cuda::vec4 dragger_inverse,
    float number_line, int brightness,
    const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by
) {
    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;



    // float distAccum = 0.0;
    Cuda::vec2 point = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);
    uint32_t dragger_color = 0;

    float dragger_lerp = 0.0;
    if (dragger_type > 0){
        Cuda::vec2 dragger_delta = point - dragger_pos;

        if (dragger_type == 2){
            dragger_delta =  Cuda::vec2(dragger_delta.x*0.71-dragger_delta.y*0.71,dragger_delta.x*0.71+dragger_delta.y*0.71);
        }

        float max_dist = max(abs(dragger_delta.x),abs(dragger_delta.y));
        float min_dist = min(abs(dragger_delta.x),abs(dragger_delta.y));

        if (max_dist < 0.24 && min_dist < 0.06){
            dragger_color = 0xffffffff;
            dragger_color = brightness+0x00ffffff;
            dragger_lerp = dragger_brightness;

        } else if (max_dist < 0.3 && min_dist < 0.12){
            dragger_lerp = dragger_brightness;
        }

        if (dragger_lerp == 1){
            pixels[pixel.y * wh.x + pixel.x] = dragger_color;
            return;
        }
    }


    Cuda::vec2 op_output = two_d_operation(point, dragger_pos, dragger_type, dragger_inverse, number_line);

    // if (pixel.x==pixel.y && pixel.y==0){
    //     printf("%f %f",op_output.x,op_output.y);
    // }
    
    float x_dist = abs(op_output.x - round(op_output.x));
    float y_dist = abs(op_output.y - round(op_output.y));
    float x_size = abs(op_output.x);
    float y_size = abs(op_output.y);

    float pixel_color = brightness+Cuda::OKLABtoRGB(0,1,op_output.x*0.1,op_output.y*0.1);
    bool fill_pixel = false;

    if (x_size < 10 && y_size < 10){
        if (!number_line){
            fill_pixel = x_dist < 0.02;
        }
        if (!number_line || y_size < 0.5){
            fill_pixel = fill_pixel || y_dist < 0.02 || y_dist*y_dist + x_dist*x_dist < 0.01;
        }
    }

    if (fill_pixel){
        uint32_t fill_color = brightness+Cuda::OKLABtoRGB(0,1,op_output.x*0.08,op_output.y*0.08);
        pixels[pixel.y * wh.x + pixel.x] =  Cuda::colorlerp(fill_color, dragger_color, dragger_lerp);
    } else {
        pixels[pixel.y * wh.x + pixel.x] =  Cuda::colorlerp(0, dragger_color, dragger_lerp);
    }

}

extern "C" void two_d_algebra(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    Cuda::vec2 dragger_pos, 
    float dragger_type, float dragger_brightness, 
    Cuda::vec4 dragger_inverse,
    int mode,
    Cuda::vec2 channels, 
    Cuda::vec2 xx,  Cuda::vec2 xy,  Cuda::vec2 yy, 
    float number_line,
    int brightness,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by
) {


    dim3 block_size(16, 16);
    dim3 grid_size((wh.x + block_size.x - 1) / block_size.x, (wh.y + block_size.y - 1) / block_size.y);

    if (mode == 0){
        two_d_algebra_grid<<<grid_size, block_size>>>( d_pixels, wh, 

            dragger_pos, 
            dragger_type, dragger_brightness, 
            dragger_inverse,
            number_line,
            brightness << 24,
            lx_ty, rx_by );
    } else {
        two_d_smallness_graph<<<grid_size, block_size>>>( d_pixels, wh, 
            mode, channels, xx, xy, yy, brightness,
            lx_ty, rx_by );

    }

}


