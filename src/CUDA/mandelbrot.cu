#include <cuda_runtime.h>
#include <vector>
#include <glm/glm.hpp>
#include <complex>
#include <cuComplex.h>  // Use cuDoubleComplex for complex numbers in CUDA

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
    unsigned int* colors
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Initialize pixel based on x and y
    cuDoubleComplex pixel = make_cuDoubleComplex(
        4 * ((x-width/2.) / (float)height),
        4 * ((float)y / (float)height - 0.5f)
    );

    pixel = cuCmul(pixel, zoom);

    // Initialize the variables for Mandelbrot iteration
    cuDoubleComplex z        = cuCadd(seed_z, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.x, 0), pixel));
    cuDoubleComplex exponent = cuCadd(seed_x, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.y, 0), pixel));
    cuDoubleComplex c        = cuCadd(seed_c, cuCmul(make_cuDoubleComplex(pixel_parameter_multipliers.z, 0), pixel));

    int iterations = 0;

    for (; iterations < max_iterations; iterations++) {
        z = cuCadd(cuCpow(z, exponent), c);
        double r = cuCreal(z);
        int i = cuCimag(z);
        if (r*r+i*i > 4.0f) {  // Escape radius is 2, compare with |z|^2 > 4
            break;
        }
    }
    unsigned int iterColor = min(iterations*4, 255);
    colors[y * width + x] = 0xff000000 | ((iterations == max_iterations) ? 0xffffffff : (iterColor + (iterColor<<8) + (iterColor<<16)));
}

// Host function to launch the kernel
extern "C" void mandelbrot_render(
    const int width, const int height,
    const std::complex<double> seed_z, const std::complex<double> seed_x, const std::complex<double> seed_c,
    const glm::vec3 pixel_parameter_multipliers,
    const std::complex<double> zoom,
    int max_iterations,  // Pass max_iterations as a parameter
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
        max_iterations, d_colors
    );

    // Copy results back from device to host
    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_colors);
}
