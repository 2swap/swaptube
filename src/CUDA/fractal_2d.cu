#include <cuda_runtime.h>
#include <complex>
#include <cuComplex.h>
#include "complex_functions.cuh"
#include "../Host_Device_Shared/vec.h"

using namespace cuCFunc;

enum fractalModes {
    MANDELBROT_2,
    MANDELBROT_3,
    MANDELBROT_POWER,
    MANDELBROT_XSET,
    REAL_COEFF_POLY,
    COMPLEX_COEFF_POLY,
    COMPLEX_C_COEFF_POLY
};

__device__ __forceinline__ float calculateParameter(const float origin, const float x_coeff, const float y_coeff, const float x, const float y){
    return origin + x_coeff * x + y_coeff * y;
}

__device__ int fractal_iterations(
    // Origin parameters
    const float zrO, const float ziO,
    const float a1rO, const float a1iO, const float ac1rO, const float ac1iO, const float x1rO, const float x1iO, 
    const float a2rO, const float a2iO, const float ac2rO, const float ac2iO, const float x2rO, const float x2iO, 
    const float a3rO, const float a3iO, const float ac3rO, const float ac3iO, const float x3rO, const float x3iO, 
    const float a4rO, const float a4iO, const float ac4rO, const float ac4iO, const float x4rO, const float x4iO,
    const float crO, const float ciO,
    // X pixel parametersW
    const float zrX, const float ziX,
    const float a1rX, const float a1iX, const float ac1rX, const float ac1iX, const float x1rX, const float x1iX, 
    const float a2rX, const float a2iX, const float ac2rX, const float ac2iX, const float x2rX, const float x2iX, 
    const float a3rX, const float a3iX, const float ac3rX, const float ac3iX, const float x3rX, const float x3iX, 
    const float a4rX, const float a4iX, const float ac4rX, const float ac4iX, const float x4rX, const float x4iX,
    const float crX, const float ciX,
    // Y pixel parameters
    const float zrY, const float ziY,
    const float a1rY, const float a1iY, const float ac1rY, const float ac1iY, const float x1rY, const float x1iY, 
    const float a2rY, const float a2iY, const float ac2rY, const float ac2iY, const float x2rY, const float x2iY, 
    const float a3rY, const float a3iY, const float ac3rY, const float ac3iY, const float x3rY, const float x3iY, 
    const float a4rY, const float a4iY, const float ac4rY, const float ac4iY, const float x4rY, const float x4iY,
    const float crY, const float ciY,
    // Other
    const int param_mode,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const float x, const float y,
    const char burning = 0, const char conj = 0
){
    switch(param_mode){
        case MANDELBROT_2:
            return cuCFunc::mandelbrot_iterations_2(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case MANDELBROT_3:
            return cuCFunc::mandelbrot_iterations_3(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case MANDELBROT_POWER:
            return cuCFunc::mandelbrot_iterations(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(x1rO, x1rX, x1rY, x, y), 
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case MANDELBROT_XSET:
            return cuCFunc::mandelbrot_iterations(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(x1rO, x1rX, x1rY, x, y), calculateParameter(x1iO, x1iX, x1iY, x, y), 
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case REAL_COEFF_POLY:
            return cuCFunc::mandelRealPoly_iterations(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(a1rO, a1rX, a1rY, x, y), calculateParameter(x1rO, x1rX, x1rY, x, y),
                calculateParameter(a2rO, a2rX, a2rY, x, y), calculateParameter(x2rO, x2rX, x2rY, x, y),
                calculateParameter(a3rO, a3rX, a3rY, x, y), calculateParameter(x3rO, x3rX, x3rY, x, y),
                calculateParameter(a4rO, a4rX, a4rY, x, y), calculateParameter(x4rO, x4rX, x4rY, x, y),
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case COMPLEX_COEFF_POLY:
            return cuCFunc::mandelPoly_iterations(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(a1rO, a1rX, a1rY, x, y), calculateParameter(a1iO, a1iX, a1iY, x, y), calculateParameter(x1rO, x1rX, x1rY, x, y), calculateParameter(x1iO, x1iX, x1iY, x, y),
                calculateParameter(a2rO, a2rX, a2rY, x, y), calculateParameter(a2iO, a2iX, a2iY, x, y), calculateParameter(x2rO, x2rX, x2rY, x, y), calculateParameter(x2iO, x2iX, x2iY, x, y),
                calculateParameter(a3rO, a3rX, a3rY, x, y), calculateParameter(a3iO, a3iX, a3iY, x, y), calculateParameter(x3rO, x3rX, x3rY, x, y), calculateParameter(x3iO, x3iX, x3iY, x, y),
                calculateParameter(a4rO, a4rX, a4rY, x, y), calculateParameter(a4iO, a4iX, a4iY, x, y), calculateParameter(x4rO, x4rX, x4rY, x, y), calculateParameter(x4iO, x4iX, x4iY, x, y),
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
                max_iterations, bailout_radius_sq, sq_radius, burning, conj);
        case COMPLEX_C_COEFF_POLY:
            return cuCFunc::mandelPolyC_iterations(
                calculateParameter(zrO, zrX, zrY, x, y), calculateParameter(ziO, ziX, ziY, x, y), 
                calculateParameter(a1rO, a1rX, a1rY, x, y), calculateParameter(a1iO, a1iX, a1iY, x, y), calculateParameter(ac1rO, ac1rX, ac1rY, x, y), calculateParameter(ac1iO, ac1iX, ac1iY, x, y), calculateParameter(x1rO, x1rX, x1rY, x, y), calculateParameter(x1iO, x1iX, x1iY, x, y),
                calculateParameter(a2rO, a2rX, a2rY, x, y), calculateParameter(a2iO, a2iX, a2iY, x, y), calculateParameter(ac2rO, ac2rX, ac2rY, x, y), calculateParameter(ac2iO, ac2iX, ac2iY, x, y), calculateParameter(x2rO, x2rX, x2rY, x, y), calculateParameter(x2iO, x2iX, x2iY, x, y),
                calculateParameter(a3rO, a3rX, a3rY, x, y), calculateParameter(a3iO, a3iX, a3iY, x, y), calculateParameter(ac3rO, ac3rX, ac3rY, x, y), calculateParameter(ac3iO, ac3iX, ac3iY, x, y), calculateParameter(x3rO, x3rX, x3rY, x, y), calculateParameter(x3iO, x3iX, x3iY, x, y),
                calculateParameter(a4rO, a4rX, a4rY, x, y), calculateParameter(a4iO, a4iX, a4iY, x, y), calculateParameter(ac4rO, ac4rX, ac4rY, x, y), calculateParameter(ac4iO, ac4iX, ac4iY, x, y), calculateParameter(x4rO, x4rX, x4rY, x, y), calculateParameter(x4iO, x4iX, x4iY, x, y),
                calculateParameter(crO, crX, crY, x, y), calculateParameter(ciO, ciX, ciY, x, y), 
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

__global__ void go(
    const int width, const int height,
    // Origin parameters
    const float zrO, const float ziO,
    const float a1rO, const float a1iO, const float ac1rO, const float ac1iO, const float x1rO, const float x1iO, 
    const float a2rO, const float a2iO, const float ac2rO, const float ac2iO, const float x2rO, const float x2iO, 
    const float a3rO, const float a3iO, const float ac3rO, const float ac3iO, const float x3rO, const float x3iO, 
    const float a4rO, const float a4iO, const float ac4rO, const float ac4iO, const float x4rO, const float x4iO,
    const float crO, const float ciO,
    // X pixel parameters
    const float zrX, const float ziX,
    const float a1rX, const float a1iX, const float ac1rX, const float ac1iX, const float x1rX, const float x1iX, 
    const float a2rX, const float a2iX, const float ac2rX, const float ac2iX, const float x2rX, const float x2iX, 
    const float a3rX, const float a3iX, const float ac3rX, const float ac3iX, const float x3rX, const float x3iX, 
    const float a4rX, const float a4iX, const float ac4rX, const float ac4iX, const float x4rX, const float x4iX,
    const float crX, const float ciX,
    // Y pixel parameters
    const float zrY, const float ziY,
    const float a1rY, const float a1iY, const float ac1rY, const float ac1iY, const float x1rY, const float x1iY, 
    const float a2rY, const float a2iY, const float ac2rY, const float ac2iY, const float x2rY, const float x2iY, 
    const float a3rY, const float a3iY, const float ac3rY, const float ac3iY, const float x3rY, const float x3iY, 
    const float a4rY, const float a4iY, const float ac4rY, const float ac4iY, const float x4rY, const float x4iY,
    const float crY, const float ciY,
    // Other
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= width || pixel_y >= height) return;

    // Scaled so squares are square
    float ndc_x = ((pixel_x + 0.5f) / fminf(width, height)) * 2.0f - (width / fminf(width, height));
    float ndc_y = ((pixel_y + 0.5f) / fminf(width, height)) * 2.0f - (height / fminf(width, height));

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
        param_mode,
        max_iterations, bailout_radius_sq, sq_radius,
        ndc_x, ndc_y,
        burning, conj
    );
    
    bool bailed_out = iterations < max_iterations;

    colors[pixel_y * width + pixel_x] = get_mandelbrot_color(iterations, max_iterations, bailed_out, sq_radius, log_real_part_exp);
}

extern "C" void fractal_2D_render(
    const int width, const int height,
    float o[28], // origin parameters
    float x[28], // x coordinate parameter multipliers
    float y[28], // y coordinate parameter multipliers
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors
) {
    unsigned int* d_colors;

    // Allocate memory on the device for the depth buffer
    cudaMalloc(&d_colors, width * height * sizeof(unsigned int));

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    go<<<numBlocks, threadsPerBlock>>>(
        width, height,
        o[0 ], o[1 ], 
        o[2 ], o[3 ], o[4 ], o[5 ], o[6 ], o[7 ],
        o[8 ], o[9 ], o[10], o[11], o[12], o[13],
        o[14], o[15], o[16], o[17], o[18], o[19],
        o[20], o[21], o[22], o[23], o[24], o[25],
        o[26], o[27],
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
        burning, conj, param_mode,
        max_iterations,
        d_colors
    );

    // Copy results back from device to host
    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_colors);
}
