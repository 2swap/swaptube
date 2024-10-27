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

// Kernel function to iterate over Mandelbrot set points
__global__ void iterate_function(
    const int width, const int height,
    const cuDoubleComplex seed_z, const cuDoubleComplex seed_x, const cuDoubleComplex seed_c,
    const glm::vec3 pixel_parameter_multipliers,
    const cuDoubleComplex zoom,
    int max_iterations,
    float gradation,
    unsigned int* colors
) {
    const unsigned int color_palette[] = {
        0xff5d0e41,
        0xff00224d,
        0xff000000,
    };
    const int palette_size = sizeof(color_palette) / sizeof(color_palette[0]);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Initialize pixel based on x and y
    cuDoubleComplex pixel = make_cuDoubleComplex(
        4 * ((x-width/2.) / (float)height),
        4 * ((float)y / (float)height - 0.5f)
    );

    pixel = cuCmul(pixel, zoom);

    cuDoubleComplex z        = cuCadd(seed_z, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.x, 0), pixel));
    cuDoubleComplex exponent = cuCadd(seed_x, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.y, 0), pixel));
    cuDoubleComplex c        = cuCadd(seed_c, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.z, 0), pixel));
    double rpe = cuCreal(exponent);
    double log_real_part_exp = log(rpe);

    double iterations = 0;
    bool bailed_out = false;

    for (; iterations < max_iterations; iterations++) {
        z = cuCadd(cuCpow(z, exponent), c);
        double r = cuCreal(z);
        double i = cuCimag(z);
        double sq_radius = r*r+i*i;
        if (sq_radius > bailout_radius_sq) {
            bailed_out = true;
            if(gradation > 0.01){
                double log_zn = log(sq_radius)/2;
                double nu = log(log_zn / log_real_part_exp) / log_real_part_exp;
                iterations += (1-nu) * gradation; // Do not use gradient for exponential parameterization
            }
            break;
        }
    }

    unsigned int internal_color = 0xffffffff;
    unsigned int color = internal_color;
    if (bailed_out) {
        int idx = floor(iterations);
        double w = iterations - idx;
        idx = (idx + 500000) % palette_size;
        color = cuda_color_lerp(color_palette[idx], color_palette[(idx+1)%palette_size], w);
        double iterfrac = iterations/max_iterations;
        iterfrac = 1-(1-iterfrac)*(1-iterfrac)*(1-iterfrac);
        color = cuda_color_lerp(color, internal_color, max(0.0, iterfrac));
    }
    colors[y * width + x] = color;
}

// Host function to launch the kernel
extern "C" void mandelbrot_render(
    const int width, const int height,
    const std::complex<double> seed_z, const std::complex<double> seed_x, const std::complex<double> seed_c,
    const glm::vec3 pixel_parameter_multipliers,
    const std::complex<double> zoom,
    int max_iterations,  // Pass max_iterations as a parameter
    float gradation,
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
    iterate_function<<<numBlocks, threadsPerBlock>>>(
        width, height,
        make_cuDoubleComplex(seed_z.real(), seed_z.imag()), make_cuDoubleComplex(seed_x.real(), seed_x.imag()), make_cuDoubleComplex(seed_c.real(), seed_c.imag()),
        pixel_parameter_multipliers,
        make_cuDoubleComplex(zoom.real(), zoom.imag()),
        max_iterations, gradation, d_colors
    );

    // Copy results back from device to host
    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_colors);
}
