#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include <complex>
#include "../Host_Device_Shared/helpers.h"
#include <cuComplex.h>  // Use cuComplex for complex numbers in CUDA

namespace Cuda {

const float bailout_radius = 256;
const float bailout_radius_sq = bailout_radius*bailout_radius;

// Function to linearly interpolate between two colors
__device__ unsigned int cuda_color_lerp(unsigned int c1, unsigned int c2, float t) {
    return ((unsigned int)((1 - t) * ((c1 >> 24) & 0xff) + t * ((c2 >> 24) & 0xff)) << 24) |
           ((unsigned int)((1 - t) * ((c1 >> 16) & 0xff) + t * ((c2 >> 16) & 0xff)) << 16) |
           ((unsigned int)((1 - t) * ((c1 >> 8 ) & 0xff) + t * ((c2 >> 8 ) & 0xff)) << 8 ) |
           ((unsigned int)((1 - t) * ( c1        & 0xff) + t * ( c2        & 0xff))      ) ;
}

__device__ cuComplex cuCpow(cuComplex base, cuComplex exponent) {
    float a = cuCrealf(base);
    float b = cuCimagf(base);
    float c = cuCrealf(exponent);
    float d = cuCimagf(exponent);
    if (a == 0.0 && b == 0.0)
        return make_cuComplex(0.0, 0.0);  // Zero raised to positive power is zero
    
    float r = sqrt(a * a + b * b);  // Magnitude of the base
    float theta = atan2(b, a);      // Argument of the base

    float new_r = pow(r, c) * exp(-d * theta);
    float new_theta = c * theta + d * log(r);

    return make_cuComplex(new_r * cos(new_theta), new_r * sin(new_theta));
}

// Color interpolation function (shared)
__device__ unsigned int get_mandelbrot_color(float iterations, int max_iterations, bool bailed_out, float gradation, float sq_radius, float log_real_part_exp, float phase_shift, unsigned int internal_color) {
    if(!bailed_out) return internal_color;

    if(bailed_out && gradation > 0.01){
        float log_zn = log(sq_radius)/2;
        float nu = log(log_zn / log_real_part_exp) / log_real_part_exp;
        iterations += (1-nu) * gradation; // Do not use gradient for exponential parameterization
    }

    const unsigned int color_palette[] = {
        0xffffffff,
        0xff000088,
        0xff000000,
        0xff000088,
    };
    const int palette_size = sizeof(color_palette) / sizeof(color_palette[0]);

    float sharpness = 15.0;
    phase_shift = fmod(phase_shift, sharpness * palette_size);
    iterations = (iterations + 100 - phase_shift) / sharpness;
    int idx = floor(iterations);
    float w = iterations - idx;
    idx %= palette_size;
    return cuda_color_lerp(color_palette[idx], color_palette[(idx + 1) % palette_size], w);
}

__device__ void compute_z_x_c(
    const vec2 pixel,
    const vec2 wh,
    const vec2 lx_ty,
    const vec2 rx_by,
    const cuComplex seed_z, const cuComplex seed_x, const cuComplex seed_c,
    const vec3 pixel_parameter_multipliers,
    cuComplex& z, cuComplex& x, cuComplex& c, float& log_real_part_exp
) {
    // Calculate the complex point based on pixel coordinates
    vec2 point_vec = pixel_to_point(pixel, lx_ty, rx_by, wh);
    cuComplex point = make_cuComplex(point_vec.x, point_vec.y);

    // Compute z, x, and c based on seed values and multipliers
    cuComplex param_complex_x = make_cuComplex(pixel_parameter_multipliers.x, 0);
    cuComplex param_complex_y = make_cuComplex(pixel_parameter_multipliers.y, 0);
    cuComplex param_complex_z = make_cuComplex(pixel_parameter_multipliers.z, 0);
    cuComplex inv_param_complex_x = make_cuComplex(1-pixel_parameter_multipliers.x, 0);
    cuComplex inv_param_complex_y = make_cuComplex(1-pixel_parameter_multipliers.y, 0);
    cuComplex inv_param_complex_z = make_cuComplex(1-pixel_parameter_multipliers.z, 0);
    z = cuCaddf(cuCmulf(inv_param_complex_x, seed_z), cuCmulf(param_complex_x, point));
    x = cuCaddf(cuCmulf(inv_param_complex_y, seed_x), cuCmulf(param_complex_y, point));
    c = cuCaddf(cuCmulf(inv_param_complex_z, seed_c), cuCmulf(param_complex_z, point));
    log_real_part_exp = log(cuCrealf(x));
}

__device__ int mandelbrot_iterations(
    cuComplex &z, const cuComplex &x, const cuComplex &c,
    int max_iterations, float bailout_radius_sq, float &sq_radius
) {
    int iterations = 0;
    sq_radius = 0;
    
    for (; iterations < max_iterations; iterations++) {
        z = cuCaddf(cuCpow(z, x), c);
        float r = cuCrealf(z);
        float i = cuCimagf(z);
        sq_radius = r * r + i * i;
        if (sq_radius > bailout_radius_sq) {
            return iterations; // Returns immediately if bailout occurs
        }
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ int mandelbrot_iterations_2or3(
    cuComplex &z, int exponent, const cuComplex &c,
    int max_iterations, float bailout_radius_sq, float &sq_radius
) {
    int iterations = 0;
    sq_radius = 0;

    // Extract real and imaginary parts of z and c
    float zr = cuCrealf(z);
    float zi = cuCimagf(z);
    float cr = cuCrealf(c);
    float ci = cuCimagf(c);

    if(exponent == 2){
        for (; iterations < max_iterations; iterations++) {
            float zr_new = zr * zr - zi * zi + cr;  // Real part of z^2 + c
            float zi_new = 2.0 * zr * zi + ci;      // Imaginary part of z^2 + c

            // Update z and square radius for next iteration
            zr = zr_new;
            zi = zi_new;
            sq_radius = zr * zr + zi * zi;

            if (sq_radius > bailout_radius_sq) return iterations;
        }
    } else {
        for (; iterations < max_iterations; iterations++) {
            float zr_new = zr * zr * zr - 3.0 * zr * zi * zi + cr;  // Real part of z^3 + c
            float zi_new = 3.0 * zr * zr * zi - zi * zi * zi + ci;  // Imaginary part of z^3 + c

            // Update z and square radius for next iteration
            zr = zr_new;
            zi = zi_new;
            sq_radius = zr * zr + zi * zi;

            if (sq_radius > bailout_radius_sq) return iterations;
        }
    }

    return max_iterations; // No bailout, maximum iterations reached
}

__global__ void go(
    const int width, const int height,
    const vec2 lx_ty,
    const vec2 rx_by,
    const cuComplex seed_z, const cuComplex seed_x, const cuComplex seed_c,
    const vec3 pixel_parameter_multipliers,
    int max_iterations,
    float gradation,
    float phase_shift,
    unsigned int internal_color,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= width || pixel_y >= height) return;
    vec2 pixel(pixel_x, pixel_y);

    cuComplex z, x, c; 
    float log_real_part_exp, sq_radius = 0;
    vec2 wh(width, height);
    compute_z_x_c(pixel, wh, lx_ty, rx_by, seed_z, seed_x, seed_c, pixel_parameter_multipliers, z, x, c, log_real_part_exp);

    // Check if the exponent 'x' is a positive integer
    bool x_is_real = (cuCimagf(x) == 0) && (cuCrealf(x) > 0) && (cuCrealf(x) == (int)cuCrealf(x));
    int intx = cuCrealf(x);

    int iterations;
    if (x_is_real && (intx == 2 || intx == 3)) {
        iterations = mandelbrot_iterations_2or3(z, intx, c, max_iterations, bailout_radius_sq, sq_radius);
    } else {
        iterations = mandelbrot_iterations(z, x, c, max_iterations, bailout_radius_sq, sq_radius);
    }
    
    bool bailed_out = iterations < max_iterations;

    colors[pixel_y * width + pixel_x] = get_mandelbrot_color(iterations, max_iterations, bailed_out, gradation, sq_radius, log_real_part_exp, phase_shift, internal_color);
}

// Host function to launch the kernel
extern "C" void mandelbrot_render(
    const int width, const int height,
    const vec2 lx_ty,
    const vec2 rx_by,
    const std::complex<float> seed_z, const std::complex<float> seed_x, const std::complex<float> seed_c,
    const vec3 pixel_parameter_multipliers,
    int max_iterations,  // Pass max_iterations as a parameter
    float gradation,
    float phase_shift,
    unsigned int internal_color,
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
        width, height, lx_ty, rx_by,
        make_cuComplex(seed_z.real(), seed_z.imag()), make_cuComplex(seed_x.real(), seed_x.imag()), make_cuComplex(seed_c.real(), seed_c.imag()),
        pixel_parameter_multipliers,
        max_iterations, gradation, phase_shift, internal_color, d_colors
    );

    // Copy results back from device to host
    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_colors);
}

}
