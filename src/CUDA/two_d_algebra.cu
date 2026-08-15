

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"




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



__global__ void two_render_real_valued_function(
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
    // float dragger_lerp = 0.0;
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
        // float dragger_dist = dragger_delta.x*dragger_delta.x + dragger_delta.y*dragger_delta.y;

        if (max_dist < 0.24 && min_dist < 0.06){
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
    
    // float minDist = 0.5;
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

    // float whiteness = max(0.0, 1.0f - distAccum*16.0);
    // uint32_t color = Cuda::colorlerp(0x00000000, Cuda::OKLABtoRGB(0,1,op_output.x*0.1,op_output.y*0.1), whiteness);
    // pixels[pixel.y * wh.x + pixel.x] =  brightness+Cuda::colorlerp(color,dragger_border,dragger_lerp);
    // pixels[pixel.y * wh.x + pixel.x] = 0xff000000 | (color << 16) | (color << 8) | color;
}

extern "C" void two_d_algebra(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    Cuda::vec2 dragger_pos, 
    float dragger_type, float dragger_brightness, 
    // float algebra,
    Cuda::vec4 dragger_inverse,
    float number_line,
    int brightness,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by
) {


    dim3 block_size(16, 16);
    dim3 grid_size((wh.x + block_size.x - 1) / block_size.x, (wh.y + block_size.y - 1) / block_size.y);
    two_render_real_valued_function<<<grid_size, block_size>>>( d_pixels, wh, 

        dragger_pos, 
        dragger_type, dragger_brightness, 
        dragger_inverse,
        number_line,
        brightness,
        lx_ty, rx_by );

}


