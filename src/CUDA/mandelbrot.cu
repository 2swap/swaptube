#include <cuda_runtime.h>
#include <vector>
#include <glm/glm.hpp>
#include <complex>
#include <cuComplex.h>  // Use cuDoubleComplex for complex numbers in CUDA

const double bailout_radius = 256;
const double bailout_radius_sq = bailout_radius*bailout_radius;

// Function to linearly interpolate between two colors
__device__ unsigned int cuda_color_lerp(unsigned int c1, unsigned int c2, double t) {
    return ((unsigned int)((1 - t) * ((c1 >> 24) & 0xff) + t * ((c2 >> 24) & 0xff)) << 24) |
           ((unsigned int)((1 - t) * ((c1 >> 16) & 0xff) + t * ((c2 >> 16) & 0xff)) << 16) |
           ((unsigned int)((1 - t) * ((c1 >> 8 ) & 0xff) + t * ((c2 >> 8 ) & 0xff)) << 8 ) |
           ((unsigned int)((1 - t) * ( c1        & 0xff) + t * ( c2        & 0xff))      ) ;
}

__device__ cuDoubleComplex cuCpow(cuDoubleComplex base, cuDoubleComplex exponent) {
    double a = cuCreal(base);
    double b = cuCimag(base);
    double c = cuCreal(exponent);
    double d = cuCimag(exponent);
    if (a == 0.0 && b == 0.0)
        return make_cuDoubleComplex(0.0, 0.0);  // Zero raised to positive power is zero
    
    double r = sqrt(a * a + b * b);  // Magnitude of the base
    double theta = atan2(b, a);      // Argument of the base

    double new_r = pow(r, c) * exp(-d * theta);
    double new_theta = c * theta + d * log(r);

    return make_cuDoubleComplex(new_r * cos(new_theta), new_r * sin(new_theta));
}

// Color interpolation function (shared)
__device__ unsigned int get_mandelbrot_color(double iterations, int max_iterations, bool bailed_out, double gradation, double sq_radius, double log_real_part_exp, double phase_shift, unsigned int internal_color) {
    if(!bailed_out) return internal_color;

    if(bailed_out && gradation > 0.01){
        double log_zn = log(sq_radius)/2;
        double nu = log(log_zn / log_real_part_exp) / log_real_part_exp;
        iterations += (1-nu) * gradation; // Do not use gradient for exponential parameterization
    }

    /* const unsigned int color_palette[] = {
        0xffffffff,
        0xffff2e63,
        0xffffffff,
        0xff08d9d6,
    }; Pastel red and blue and black POP */
    /* const unsigned int color_palette[] = {
        0xffffffff,
        0xffff33fc,
        0xff5d21a7,
        0xff1571df,
        0xff25f7ff,
    }; Used in the mandelbrot thumbnail */
    const unsigned int color_palette[] = {
        0xffffffff,
        0xff000088,
        0xff000000,
        0xff000088,
    };
    const int palette_size = sizeof(color_palette) / sizeof(color_palette[0]);

    iterations = (iterations + 100 - phase_shift) / 15.;
    int idx = floor(iterations);
    double w = iterations - idx;
    idx %= palette_size;
    return cuda_color_lerp(color_palette[idx], color_palette[(idx + 1) % palette_size], w);
}

__device__ void compute_z_x_c(
    int pixel_x, int pixel_y, int width, int height,
    const cuDoubleComplex seed_z, const cuDoubleComplex seed_x, const cuDoubleComplex seed_c,
    const glm::dvec3 pixel_parameter_multipliers,
    const cuDoubleComplex zoom,
    cuDoubleComplex& z, cuDoubleComplex& x, cuDoubleComplex& c, double& log_real_part_exp
) {
    // Calculate the complex point based on pixel coordinates
    cuDoubleComplex point = make_cuDoubleComplex(
        4 * ((pixel_x - width / 2.0) / static_cast<float>(height)),
        4 * ((static_cast<float>(pixel_y) / height) - 0.5f)
    );

    point = cuCmul(point, zoom);

    // Compute z, x, and c based on seed values and multipliers
    z = cuCadd(seed_z, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.x, 0), point));
    x = cuCadd(seed_x, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.y, 0), point));
    c = cuCadd(seed_c, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.z, 0), point));
    double rpe = cuCreal(x);
    log_real_part_exp = log(rpe);
}

__device__ int mandelbrot_iterations(
    cuDoubleComplex &z, const cuDoubleComplex &x, const cuDoubleComplex &c,
    int max_iterations, double bailout_radius_sq, double &sq_radius
) {
    int iterations = 0;
    sq_radius = 0;
    
    for (; iterations < max_iterations; iterations++) {
        z = cuCadd(cuCpow(z, x), c);
        double r = cuCreal(z);
        double i = cuCimag(z);
        sq_radius = r * r + i * i;
        if (sq_radius > bailout_radius_sq) {
            return iterations; // Returns immediately if bailout occurs
        }
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ int mandelbrot_iterations_2or3(
    cuDoubleComplex &z, int exponent, const cuDoubleComplex &c,
    int max_iterations, double bailout_radius_sq, double &sq_radius
) {
    int iterations = 0;
    sq_radius = 0;

    // Extract real and imaginary parts of z and c
    double zr = cuCreal(z);
    double zi = cuCimag(z);
    double cr = cuCreal(c);
    double ci = cuCimag(c);

    if(exponent == 2){
        for (; iterations < max_iterations; iterations++) {
            double zr_new = zr * zr - zi * zi + cr;  // Real part of z^2 + c
            double zi_new = 2.0 * zr * zi + ci;      // Imaginary part of z^2 + c

            // Update z and square radius for next iteration
            zr = zr_new;
            zi = zi_new;
            sq_radius = zr * zr + zi * zi;

            if (sq_radius > bailout_radius_sq) return iterations;
        }
    } else {
        for (; iterations < max_iterations; iterations++) {
            double zr_new = zr * zr * zr - 3.0 * zr * zi * zi + cr;  // Real part of z^3 + c
            double zi_new = 3.0 * zr * zr * zi - zi * zi * zi + ci;  // Imaginary part of z^3 + c

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
    const cuDoubleComplex seed_z, const cuDoubleComplex seed_x, const cuDoubleComplex seed_c,
    const glm::dvec3 pixel_parameter_multipliers,
    const cuDoubleComplex zoom,
    int max_iterations,
    float gradation,
    float phase_shift,
    unsigned int internal_color,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= width || pixel_y >= height) return;

    cuDoubleComplex z, x, c; 
    double log_real_part_exp, sq_radius = 0;
    compute_z_x_c(pixel_x, pixel_y, width, height, seed_z, seed_x, seed_c, pixel_parameter_multipliers, zoom, z, x, c, log_real_part_exp);

    // Check if the exponent 'x' is a positive integer
    bool x_is_real = (cuCimag(x) == 0) && (cuCreal(x) > 0) && (cuCreal(x) == (int)cuCreal(x));
    int intx = cuCreal(x);

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
    const std::complex<double> seed_z, const std::complex<double> seed_x, const std::complex<double> seed_c,
    const glm::dvec3 pixel_parameter_multipliers,
    const std::complex<double> zoom,
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
        width, height,
        make_cuDoubleComplex(seed_z.real(), seed_z.imag()), make_cuDoubleComplex(seed_x.real(), seed_x.imag()), make_cuDoubleComplex(seed_c.real(), seed_c.imag()),
        pixel_parameter_multipliers,
        make_cuDoubleComplex(zoom.real(), zoom.imag()),
        max_iterations, gradation, phase_shift, internal_color, d_colors
    );

    // Copy results back from device to host
    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_colors);
}
