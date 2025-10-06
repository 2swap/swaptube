#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../Host_Device_Shared/helpers.h"
#include "color.cuh"

__device__ thrust::complex<float> complex_sqrt(const thrust::complex<float>& z, float branch_cut) {
    float r = thrust::abs(z);
    float theta = atan2(z.imag(), z.real());
    if (theta < branch_cut) {
        theta += 2 * M_PI; // Adjust angle based on branch cut
    }
    return thrust::polar(sqrt(r), theta / 2);
}

__global__ void render_kernel(
    int* d_pixels,
    glm::vec2 wh,
    float sqrt_coef, float sqrt_branch_cut, float sin_coef, float cos_coef, float exp_coef,
    glm::vec2 lx_ty,
    glm::vec2 rx_by,
    float ab_dilation,
    float dot_radius
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= wh.x || y >= wh.y) return;

    const glm::vec2 point = pixel_to_point(glm::vec2(x,y), lx_ty, rx_by, wh);
    thrust::complex<float> val;
    const thrust::complex<float> sqrt_val = complex_sqrt(val, sqrt_branch_cut);
    const thrust::complex<float> sin_val = thrust::sin(val);
    const thrust::complex<float> cos_val = thrust::cos(val);
    const thrust::complex<float> exp_val = thrust::exp(val);
    const thrust::complex<float> sum = sqrt_coef * sqrt_val + sin_coef * sin_val + cos_coef * cos_val + exp_coef * exp_val;
    const int color = d_complex_to_srgb(sum, ab_dilation, dot_radius);

    d_pixels[y * int(wh.x) + x] = color;
}

extern "C" void color_complex_arbitrary_function(
    unsigned int* h_pixels, // to be overwritten with the result
    int w,
    int h,
    float sqrt_coef, float sqrt_branch_cut, float sin_coef, float cos_coef, float exp_coef,
    float lx, float ty,
    float rx, float by,
    float ab_dilation,
    float dot_radius
) {
    // Allocate device memory for pixels
    int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(int));

    // Define the region in complex plane
    glm::vec2 wh(w, h);
    glm::vec2 lx_ty(lx, ty);
    glm::vec2 rx_by(rx, by);

    // Kernel config
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    render_kernel<<<gridSize, blockSize>>>(d_pixels, wh, sqrt_coef, sqrt_branch_cut, sin_coef, cos_coef, exp_coef, lx_ty, rx_by, ab_dilation, dot_radius);
    cudaDeviceSynchronize();

    // Copy pixels back to host
    cudaMemcpy(h_pixels, d_pixels, w * h * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
}
