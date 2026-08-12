#include <cuda_runtime.h>
#include <complex>
#include <cuComplex.h>
#include "complex_functions.cuh"
#include "common_graphics.cuh"
#include "../Host_Device_Shared/vec.h"

enum fractalModes {
    MANDELBROT_2,
    MANDELBROT_3,
    MANDELBROT_POWER,
    MANDELBROT_XSET,
    REAL_COEFF_POLY,
    COMPLEX_COEFF_POLY,
    COMPLEX_C_COEFF_POLY
};

__device__ __forceinline__ float calculateParameter(const float origin, const float x_macro_coeff, const float y_macro_coeff, const float x_micro_coeff, const float y_micro_coeff, const float macro_x, const float macro_y, const float micro_x, const float micro_y) {
    return origin + x_macro_coeff * macro_x + y_macro_coeff * macro_y + x_micro_coeff * micro_x + y_micro_coeff * micro_y;
}

__device__ int fractal_iterations(
    // Origin parameters
    const float zrO, const float ziO,
    const float a1rO, const float a1iO, const float ac1rO, const float ac1iO, const float x1rO, const float x1iO, 
    const float a2rO, const float a2iO, const float ac2rO, const float ac2iO, const float x2rO, const float x2iO, 
    const float a3rO, const float a3iO, const float ac3rO, const float ac3iO, const float x3rO, const float x3iO, 
    const float a4rO, const float a4iO, const float ac4rO, const float ac4iO, const float x4rO, const float x4iO,
    const float crO, const float ciO,
    // X macro pixel parameters
    const float zrX, const float ziX,
    const float a1rX, const float a1iX, const float ac1rX, const float ac1iX, const float x1rX, const float x1iX, 
    const float a2rX, const float a2iX, const float ac2rX, const float ac2iX, const float x2rX, const float x2iX, 
    const float a3rX, const float a3iX, const float ac3rX, const float ac3iX, const float x3rX, const float x3iX, 
    const float a4rX, const float a4iX, const float ac4rX, const float ac4iX, const float x4rX, const float x4iX,
    const float crX, const float ciX,
    // Y macro pixel parameters
    const float zrY, const float ziY,
    const float a1rY, const float a1iY, const float ac1rY, const float ac1iY, const float x1rY, const float x1iY, 
    const float a2rY, const float a2iY, const float ac2rY, const float ac2iY, const float x2rY, const float x2iY, 
    const float a3rY, const float a3iY, const float ac3rY, const float ac3iY, const float x3rY, const float x3iY, 
    const float a4rY, const float a4iY, const float ac4rY, const float ac4iY, const float x4rY, const float x4iY,
    const float crY, const float ciY,
    // X micro pixel parameters
    const float zrx, const float zix,
    const float a1rx, const float a1ix, const float ac1rx, const float ac1ix, const float x1rx, const float x1ix, 
    const float a2rx, const float a2ix, const float ac2rx, const float ac2ix, const float x2rx, const float x2ix, 
    const float a3rx, const float a3ix, const float ac3rx, const float ac3ix, const float x3rx, const float x3ix, 
    const float a4rx, const float a4ix, const float ac4rx, const float ac4ix, const float x4rx, const float x4ix,
    const float crx, const float cix,
    // Y micro pixel parameters
    const float zry, const float ziy,
    const float a1ry, const float a1iy, const float ac1ry, const float ac1iy, const float x1ry, const float x1iy, 
    const float a2ry, const float a2iy, const float ac2ry, const float ac2iy, const float x2ry, const float x2iy, 
    const float a3ry, const float a3iy, const float ac3ry, const float ac3iy, const float x3ry, const float x3iy, 
    const float a4ry, const float a4iy, const float ac4ry, const float ac4iy, const float x4ry, const float x4iy,
    const float cry, const float ciy,
    // Other
    const int param_mode,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const float macro_x, const float macro_y,
    const float micro_x, const float micro_y,
    const char burning = 0, const char conj = 0
){
    switch(param_mode){
        case MANDELBROT_2:
            return mandelbrot_iterations_2(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case MANDELBROT_3:
            return mandelbrot_iterations_3(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case MANDELBROT_POWER:
            return mandelbrot_iterations(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(x1rO, x1rX, x1rY, x1rx, x1ry, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y),
                max_iterations, bailout_radius_sq, sq_radius,
                burning,
                conj);
        case MANDELBROT_XSET:
            return mandelbrot_iterations(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(x1rO, x1rX, x1rY, x1rx, x1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x1iO, x1iX, x1iY, x1ix, x1iy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case REAL_COEFF_POLY:
            return mandelRealPoly_iterations(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(a1rO, a1rX, a1rY, a1rx, a1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x1rO, x1rX, x1rY, x1rx, x1ry, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a2rO, a2rX, a2rY, a2rx, a2ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x2rO, x2rX, x2rY, x2rx, x2ry, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a3rO, a3rX, a3rY, a3rx, a3ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x3rO, x3rX, x3rY, x3rx, x3ry, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a4rO, a4rX, a4rY, a4rx, a4ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x4rO, x4rX, x4rY, x4rx, x4ry, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y),
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case COMPLEX_COEFF_POLY:
            return mandelPoly_iterations(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(a1rO, a1rX, a1rY, a1rx, a1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a1iO, a1iX, a1iY, a1ix, a1iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x1rO, x1rX, x1rY, x1rx, x1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x1iO, x1iX, x1iY, x1ix, x1iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a2rO, a2rX, a2rY, a2rx, a2ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a2iO, a2iX, a2iY, a2ix, a2iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x2rO, x2rX, x2rY, x2rx, x2ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x2iO, x2iX, x2iY, x2ix, x2iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a3rO, a3rX, a3rY, a3rx, a3ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a3iO, a3iX, a3iY, a3ix, a3iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x3rO, x3rX, x3rY, x3rx, x3ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x3iO, x3iX, x3iY, x3ix, x3iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a4rO, a4rX, a4rY, a4rx, a4ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a4iO, a4iX, a4iY, a4ix, a4iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x4rO, x4rX, x4rY, x4rx, x4ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x4iO, x4iX, x4iY, x4ix, x4iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y),
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case COMPLEX_C_COEFF_POLY:
            return mandelPolyC_iterations(
                calculateParameter(zrO, zrX, zrY, zrx, zry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ziO, ziX, ziY, zix, ziy, macro_x, macro_y, micro_x, micro_y), 
                calculateParameter(a1rO, a1rX, a1rY, a1rx, a1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a1iO, a1iX, a1iY, a1ix, a1iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac1rO, ac1rX, ac1rY, ac1rx, ac1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac1iO, ac1iX, ac1iY, ac1ix, ac1iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x1rO, x1rX, x1rY, x1rx, x1ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x1iO, x1iX, x1iY, x1ix, x1iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a2rO, a2rX, a2rY, a2rx, a2ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a2iO, a2iX, a2iY, a2ix, a2iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac2rO, ac2rX, ac2rY, ac2rx, ac2ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac2iO, ac2iX, ac2iY, ac2ix, ac2iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x2rO, x2rX, x2rY, x2rx, x2ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x2iO, x2iX, x2iY, x2ix, x2iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a3rO, a3rX, a3rY, a3rx, a3ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a3iO, a3iX, a3iY, a3ix, a3iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac3rO, ac3rX, ac3rY, ac3rx, ac3ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac3iO, ac3iX, ac3iY, ac3ix, ac3iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x3rO, x3rX, x3rY, x3rx, x3ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x3iO, x3iX, x3iY, x3ix, x3iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(a4rO, a4rX, a4rY, a4rx, a4ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(a4iO, a4iX, a4iY, a4ix, a4iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac4rO, ac4rX, ac4rY, ac4rx, ac4ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ac4iO, ac4iX, ac4iY, ac4ix, ac4iy, macro_x, macro_y, micro_x, micro_y), calculateParameter(x4rO, x4rX, x4rY, x4rx, x4ry, macro_x, macro_y, micro_x, micro_y), calculateParameter(x4iO, x4iX, x4iY, x4ix, x4iy, macro_x, macro_y, micro_x, micro_y),
                calculateParameter(crO, crX, crY, crx, cry, macro_x, macro_y, micro_x, micro_y), calculateParameter(ciO, ciX, ciY, cix, ciy, macro_x, macro_y, micro_x, micro_y),
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
    }

    return 0;
}

// to be replaced
__device__ unsigned int get_mandelbrot_color(int iterations, int max_iterations, bool bailed_out, float sq_radius, float log_real_part_exp){
    if(!bailed_out) return 0xFF000000; // black for points that didn't escape

    // Map iterations to a color gradient
    float t = (float)iterations / max_iterations;

    unsigned char c = (unsigned char)(t * 255);

    return 0xff000000 | (c << 16) | (c << 8) | c; // grayscale color
}

__device__ Cuda::vec3 bezier_gradient(Cuda::vec3 mid, float t){
    Cuda::vec3 a = t * mid;
    Cuda::vec3 b = (1.0f - t) * mid + t * Cuda::vec3(1.0, 1.0, 1.0);
    return (1.0f - t) * a + t * b;
}

__device__ uint32_t vec3_to_argb(float alpha, const Cuda::vec3& rgb){
    return ((int) (alpha * 255.0) << 24) | ((int) (rgb.x * 255.0) << 16) | ((int) (rgb.y * 255.0) << 8) | ((int) (rgb.z * 255.0));
}

__global__ void go(
    const Cuda::ivec2 wh,
    // Origin parameters
    const float zrO, const float ziO,
    const float a1rO, const float a1iO, const float ac1rO, const float ac1iO, const float x1rO, const float x1iO, 
    const float a2rO, const float a2iO, const float ac2rO, const float ac2iO, const float x2rO, const float x2iO, 
    const float a3rO, const float a3iO, const float ac3rO, const float ac3iO, const float x3rO, const float x3iO, 
    const float a4rO, const float a4iO, const float ac4rO, const float ac4iO, const float x4rO, const float x4iO,
    const float crO, const float ciO,
    // X macro pixel parameters
    const float zrX, const float ziX,
    const float a1rX, const float a1iX, const float ac1rX, const float ac1iX, const float x1rX, const float x1iX, 
    const float a2rX, const float a2iX, const float ac2rX, const float ac2iX, const float x2rX, const float x2iX, 
    const float a3rX, const float a3iX, const float ac3rX, const float ac3iX, const float x3rX, const float x3iX, 
    const float a4rX, const float a4iX, const float ac4rX, const float ac4iX, const float x4rX, const float x4iX,
    const float crX, const float ciX,
    // Y macro pixel parameters
    const float zrY, const float ziY,
    const float a1rY, const float a1iY, const float ac1rY, const float ac1iY, const float x1rY, const float x1iY, 
    const float a2rY, const float a2iY, const float ac2rY, const float ac2iY, const float x2rY, const float x2iY, 
    const float a3rY, const float a3iY, const float ac3rY, const float ac3iY, const float x3rY, const float x3iY, 
    const float a4rY, const float a4iY, const float ac4rY, const float ac4iY, const float x4rY, const float x4iY,
    const float crY, const float ciY,
    // X micro pixel parameters
    const float zrx, const float zix,
    const float a1rx, const float a1ix, const float ac1rx, const float ac1ix, const float x1rx, const float x1ix, 
    const float a2rx, const float a2ix, const float ac2rx, const float ac2ix, const float x2rx, const float x2ix, 
    const float a3rx, const float a3ix, const float ac3rx, const float ac3ix, const float x3rx, const float x3ix, 
    const float a4rx, const float a4ix, const float ac4rx, const float ac4ix, const float x4rx, const float x4ix,
    const float crx, const float cix,
    // Y micro pixel parameters
    const float zry, const float ziy,
    const float a1ry, const float a1iy, const float ac1ry, const float ac1iy, const float x1ry, const float x1iy, 
    const float a2ry, const float a2iy, const float ac2ry, const float ac2iy, const float x2ry, const float x2iy, 
    const float a3ry, const float a3iy, const float ac3ry, const float ac3iy, const float x3ry, const float x3iy, 
    const float a4ry, const float a4iy, const float ac4ry, const float ac4iy, const float x4ry, const float x4iy,
    const float cry, const float ciy,
    // Other
    const float sub_dimensions_x, const float sub_dimensions_y,
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    // Scaled so squares are square (coordinates from -1, -1 to 1, 1)
    //float ndc_x = ((pixel_x + 0.5f) / fminf(wh.x, wh.y)) * 2.0f - (wh.x / fminf(wh.x, wh.y));
    //float ndc_y = -(((pixel_y + 0.5f) / fminf(wh.x, wh.y)) * 2.0f - (wh.y / fminf(wh.x, wh.y)));

    float ndc_x = ((pixel_x + 1.0f) / wh.x) - 0.5f;
    float ndc_y = -(((pixel_y + 1.0f) / wh.y) - 0.5f);

    Cuda::vec2 sub_dimensions(sub_dimensions_x, sub_dimensions_y);

    Cuda::vec2 tile_size = Cuda::vec2(wh) / sub_dimensions;

    float macro_aspect = (float) wh.x / wh.y;
    float micro_aspect = (wh.x * sub_dimensions.y) / (wh.y * sub_dimensions.x);

    float scaled_x = ndc_x * sub_dimensions.x;
    float scaled_y = ndc_y * sub_dimensions.y;

    float macro_x = roundf(scaled_x);
    float macro_y = roundf(scaled_y);

    float micro_ndc_x = roundf((scaled_x - macro_x) * 2 * micro_aspect * tile_size.x) / tile_size.x;
    float micro_ndc_y = roundf((scaled_y - macro_y) * 2 * tile_size.y) / tile_size.y;

    float macro_ndc_x = (macro_x) / (sub_dimensions.x * 0.5) * macro_aspect;
    float macro_ndc_y = (macro_y) / (sub_dimensions.y * 0.5);

    float log_real_part_exp, sq_radius = 0;
    float bailout_radius_sq = 256.0 * 256.0;

    int iterations = fractal_iterations(
        zrO, ziO,
        a1rO, a1iO, ac1rO, ac1iO, x1rO, x1iO, 
        a2rO, a2iO, ac2rO, ac2iO, x2rO, x2iO, 
        a3rO, a3iO, ac3rO, ac3iO, x3rO, x3iO, 
        a4rO, a4iO, ac4rO, ac4iO, x4rO, x4iO,
        crO, ciO,
        zrX, ziX,
        a1rX, a1iX, ac1rX, ac1iX, x1rX, x1iX, 
        a2rX, a2iX, ac2rX, ac2iX, x2rX, x2iX, 
        a3rX, a3iX, ac3rX, ac3iX, x3rX, x3iX, 
        a4rX, a4iX, ac4rX, ac4iX, x4rX, x4iX,
        crX, ciX,
        zrY, ziY,
        a1rY, a1iY, ac1rY, ac1iY, x1rY, x1iY, 
        a2rY, a2iY, ac2rY, ac2iY, x2rY, x2iY,
        a3rY, a3iY, ac3rY, ac3iY, x3rY, x3iY,
        a4rY, a4iY, ac4rY, ac4iY, x4rY, x4iY,
        crY, ciY,
        zrx, zix,
        a1rx, a1ix, ac1rx, ac1ix, x1rx, x1ix, 
        a2rx, a2ix, ac2rx, ac2ix, x2rx, x2ix, 
        a3rx, a3ix, ac3rx, ac3ix, x3rx, x3ix, 
        a4rx, a4ix, ac4rx, ac4ix, x4rx, x4ix,
        crx, cix,
        zry, ziy,
        a1ry, a1iy, ac1ry, ac1iy, x1ry, x1iy, 
        a2ry, a2iy, ac2ry, ac2iy, x2ry, x2iy,
        a3ry, a3iy, ac3ry, ac3iy, x3ry, x3iy,
        a4ry, a4iy, ac4ry, ac4iy, x4ry, x4iy,
        cry, ciy,
        param_mode,
        max_iterations, bailout_radius_sq, sq_radius,
        macro_ndc_x, macro_ndc_y,
        micro_ndc_x, micro_ndc_y,
        burning, conj
    );
    
    bool bailed_out = iterations < max_iterations;
    if(iterations == 50){
         colors[pixel_y * wh.x + pixel_x] = 0xff000000;
    }
    else{
        colors[pixel_y * wh.x + pixel_x] = 0xffffffff;//vec3_to_argb(1.0, bezier_gradient(Cuda::vec3(0.0, 0.0, 1.0), sqrtf(iterations / 50.0)));
    }
}

extern "C" void fractal_2D_render(
    const Cuda::ivec2& wh,
    float o[28], // origin parameters
    float X[28], // x coordinate macro parameter multipliers
    float Y[28], // y coordinate macro parameter multipliers
    float x[28], // x coordinate micro parameter multipliers
    float y[28], // y coordinate micro parameter multipliers
    const float sub_dimensions_x, const float sub_dimensions_y,
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    go<<<numBlocks, threadsPerBlock>>>(
        wh,
        o[0 ], o[1 ], 
        o[2 ], o[3 ], o[4 ], o[5 ], o[6 ], o[7 ],
        o[8 ], o[9 ], o[10], o[11], o[12], o[13],
        o[14], o[15], o[16], o[17], o[18], o[19],
        o[20], o[21], o[22], o[23], o[24], o[25],
        o[26], o[27],
        X[0 ], X[1 ], 
        X[2 ], X[3 ], X[4 ], X[5 ], X[6 ], X[7 ],
        X[8 ], X[9 ], X[10], X[11], X[12], X[13],
        X[14], X[15], X[16], X[17], X[18], X[19],
        X[20], X[21], X[22], X[23], X[24], X[25],
        X[26], X[27],
        Y[0 ], Y[1 ], 
        Y[2 ], Y[3 ], Y[4 ], Y[5 ], Y[6 ], Y[7 ],
        Y[8 ], Y[9 ], Y[10], Y[11], Y[12], Y[13],
        Y[14], Y[15], Y[16], Y[17], Y[18], Y[19],
        Y[20], Y[21], Y[22], Y[23], Y[24], Y[25],
        Y[26], Y[27],
        x[0 ], x[1 ], 
        x[2 ], x[3 ], x[4 ], x[5 ], x[6 ], x[7 ],
        x[8 ], x[9 ], x[10], x[11], x[12], x[13],
        x[14], x[15], x[16], x[17], x[18], x[19],
        x[20], x[21], x[22], x[23], x[24], x[25],
        x[26], x[27],
        y[0 ], y[1 ], 
        y[2 ], y[3 ], y[4 ], y[5 ], y[6 ], y[7 ],
        y[8 ], y[9 ], y[10], y[11], y[12], y[13],
        y[14], y[15], y[16], y[17], y[18], y[19],
        y[20], y[21], y[22], y[23], y[24], y[25],
        y[26], y[27],
        sub_dimensions_x, sub_dimensions_y,
        burning, conj, param_mode,
        max_iterations,
        colors
    );
}
