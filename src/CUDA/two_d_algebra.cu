

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"




__device__ Cuda::vec2 two_d_operation(
    Cuda::vec2 point, Cuda::vec2 dragger, float dragger_type, float algebra
    // Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yx, Cuda::vec2 yy
){
    if (dragger_type == 0){
        return point;

    } else if (dragger_type == 1){
        return point + dragger;

    } else if (algebra == 0){
        return Cuda::vec2(point.x*dragger.x,point.y*dragger.y);

    } else if (algebra == 1){
        // return Cuda::vec2(point.x*dragger.x,point.y*dragger.y);
        return Cuda::vec2(
            point.x*dragger.x-point.y*dragger.y,
            point.y*dragger.x+point.x*dragger.y
        );

    } else if (algebra == 2){
        // return Cuda::vec2(point.x*dragger.x,point.y*dragger.y);

        if (abs(dragger.y) < 0.01){
            if (abs(point.y-point.x) < 0.1){
                return Cuda::vec2(-point.x,-point.x);
            }
            return Cuda::vec2(10000,10000);
        }

        float y_pos = (point.y-point.x)/dragger.y;

        return Cuda::vec2(
            (-point.x-dragger.x*y_pos)/dragger.y,
            y_pos
        );

    } else if (algebra == 3){
        // return Cuda::vec2(point.x*dragger.x,point.y*dragger.y);
        return Cuda::vec2(
            point.x*dragger.x,
            point.y*dragger.x+point.x*dragger.y
        );


    } else if (algebra == 4){
        // return Cuda::vec2(point.x*dragger.x,point.y*dragger.y);
        return Cuda::vec2(
            point.x*dragger.x+point.y*dragger.y,
            point.y*dragger.x+point.x*dragger.y
        );

    // } else {
    //     return Cuda::vec2(
    //         dragger.x*point.x*xx.x + dragger.x*point.y*xy.x + dragger.y*point.x*yx.x + dragger.y*point.y*yy.x,
    //         dragger.x*point.x*xx.y + dragger.x*point.y*xy.y + dragger.y*point.x*yx.y + dragger.y*point.y*yy.y
    //     );
    }
    return Cuda::vec2(0,0);

}



__global__ void two_render_real_valued_function(
    uint32_t* pixels, const Cuda::ivec2 wh,
    // Cuda::ResolvedStateEquationComponent* xd_eq, int x_eq_size, float x_adjustment,
    // Cuda::ResolvedStateEquationComponent* yd_eq, int y_eq_size, float y_adjustment,
    Cuda::vec2 dragger, Cuda::vec2 dragger_pos, float dragger_type, float algebra,
    // Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yx, Cuda::vec2 yy, 
    float number_line, int brightness,
    const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by
) {
    Cuda::ivec2 pixel(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= wh.x || pixel.y >= wh.y) return;


    float distAccum = 0.0;
    float dragger_lerp = 0.0;
    Cuda::vec2 point = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);

    int dragger_fill = 0x00000000;
    int dragger_border = 0x00ffffff;
    if (dragger_type > 1){
        dragger_fill = 0x00ffffff;
        dragger_border = 0x00000000;
    }

    if (dragger_type > 0){

        Cuda::vec2 dragger_delta = point - dragger_pos;
        float dragger_dist = dragger_delta.x*dragger_delta.x + dragger_delta.y*dragger_delta.y;

        if (dragger_dist < 0.03){
            pixels[pixel.y * wh.x + pixel.x] = brightness+dragger_fill;
            return;
        } else if (dragger_dist < 0.05){
            pixels[pixel.y * wh.x + pixel.x] = brightness+Cuda::colorlerp(dragger_fill, dragger_border, (dragger_dist-0.03)*50);
            return;
        } else if (dragger_dist < 0.07){
            dragger_lerp = (0.07-dragger_dist)*50;
        }
    }


    // float cuda_tags[2] = { point.x, point.y };
    // int error = 0;
    // float x_eval = evaluate_resolved_state_equation(x_eq_size, xd_eq, cuda_tags, 2, error);
    // float y_eval = evaluate_resolved_state_equation(y_eq_size, yd_eq, cuda_tags, 2, error);

    Cuda::vec2 op_output = two_d_operation(point, dragger, dragger_type, algebra);

    // if (pixel.x==pixel.y && pixel.y==0){
    //     printf("%f %f",op_output.x,op_output.y);
    // }
    
    float minDist = 0.5;
    float x_dist = abs(op_output.x - round(op_output.x));
    float y_dist = abs(op_output.y - round(op_output.y));
    float x_size = abs(op_output.x);
    float y_size = abs(op_output.y);

    if (x_size < 10 && y_size < 10){

        if (!number_line ){
            minDist = min(minDist, x_dist);
        }

        if (!number_line || y_size < 0.5){
            minDist = min(minDist, y_dist);
            minDist = min(minDist, (y_dist*y_dist + x_dist*x_dist)*2.5);
        }
    }
    distAccum += minDist;


    float whiteness = max(0.0, 1.0f - distAccum*16.0);
    uint32_t color = Cuda::colorlerp(0x00000000, Cuda::OKLABtoRGB(0,1,op_output.x*0.1,op_output.y*0.1), whiteness);
    // uint32_t color = Cuda::OKLABtoRGB(255,whiteness,x_eval*0.2,y_eval*0.2);
    pixels[pixel.y * wh.x + pixel.x] =  brightness+Cuda::colorlerp(color,dragger_border,dragger_lerp);
    // pixels[pixel.y * wh.x + pixel.x] = 0xff000000 | (color << 16) | (color << 8) | color;
}

extern "C" void two_d_algebra(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    // Cuda::ResolvedStateEquationComponent* x_eq, int x_eq_size, float x_adjustment,
    // Cuda::ResolvedStateEquationComponent* y_eq, int y_eq_size, float y_adjustment,
    Cuda::vec2 dragger, Cuda::vec2 dragger_pos, float dragger_type, float algebra,
    // Cuda::vec2 xx, Cuda::vec2 xy, Cuda::vec2 yx, Cuda::vec2 yy, 
    float number_line,
    int brightness,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by
) {

    // Cuda::ResolvedStateEquationComponent* xd_eq;
    // size_t x_memsize = x_eq_size * sizeof(Cuda::ResolvedStateEquationComponent);
    // cudaMalloc(&xd_eq, x_memsize);
    // cudaMemcpy(xd_eq, x_eq, x_memsize, cudaMemcpyHostToDevice);

    // Cuda::ResolvedStateEquationComponent* yd_eq;
    // size_t y_memsize = y_eq_size * sizeof(Cuda::ResolvedStateEquationComponent);
    // cudaMalloc(&yd_eq, y_memsize);
    // cudaMemcpy(yd_eq, y_eq, y_memsize, cudaMemcpyHostToDevice);


    dim3 block_size(16, 16);
    dim3 grid_size((wh.x + block_size.x - 1) / block_size.x, (wh.y + block_size.y - 1) / block_size.y);
    two_render_real_valued_function<<<grid_size, block_size>>>( d_pixels, wh, 
        // xd_eq, x_eq_size, x_adjustment,
        // yd_eq, y_eq_size, y_adjustment,
        dragger, dragger_pos, dragger_type, algebra,
        // xx, xy, yx, yy,
        number_line,
        brightness,
        lx_ty, rx_by );

    // cudaFree(xd_eq);
    // cudaFree(yd_eq);
}


