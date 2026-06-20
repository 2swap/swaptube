#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdio>
#include "common_graphics.cuh"
#include "find_roots.cuh"
#include "../Host_Device_Shared/helpers.h"

__device__ cuFloatComplex complex_pow(cuFloatComplex z, int n) {
    cuFloatComplex result = make_cuFloatComplex(1.0f, 0.0f);
    for (int i = 0; i < n; i++) {
        result = cuCmulf(result, z);
    }
    return result;
}

// Make a circular gradient on a pixel buffer
__device__ void d_gradient_circle(float cx, float cy, float radius, float red, float green, float blue,
        float* d_alpha, float* d_red, float* d_green, float* d_blue, const Cuda::ivec2 wh, float opa=1.0f) {
    // breakout if any part of the circle is outside of screen
    if (cx < 0 || cx >= wh.x || cy < 0 || cy >= wh.y)
        return;
    float radius2 = radius * radius;
    for (int x = cx - radius; x < cx + radius; x++) {
        float sdx = (x - cx)*(x - cx);
        for (int y = cy - radius; y < cy + radius; y++) {
            float sdy = (y - cy)*(y - cy);
            float dist2 = (sdx + sdy) / radius2;
            if (dist2 < 1.0f) {
                float final_opa = opa / (.025 + 160 * dist2 * dist2);
                if (x >= 0 && x < wh.x && y >= 0 && y < wh.y) {
                    atomicAdd(&d_alpha[y * wh.x + x],         final_opa);
                    atomicAdd(&d_red  [y * wh.x + x], red   * final_opa);
                    atomicAdd(&d_green[y * wh.x + x], green * final_opa);
                    atomicAdd(&d_blue [y * wh.x + x], blue  * final_opa);
                }
            }
        }
    }
}

__global__ void root_fractal_kernel(float* d_alpha, float* d_red, float* d_green, float* d_blue, const Cuda::ivec2 wh, cuFloatComplex c1, cuFloatComplex c2, float terms, const Cuda::vec2 lx_ty, const Cuda::vec2 rx_by, float radius, float opacity) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ceil_terms = ceil(terms);
    unsigned int total = 1 << ceil_terms; // total number of polynomials

    if (idx >= total) return;

    cuFloatComplex coeffs[30];
    float red = 0.0f, green = 0.0f, blue = 0.0f;
    for (int i = 0; i < ceil_terms; i++) {
        // Set Coefficient
        bool bit = (idx >> i) & 1;
        coeffs[i] = bit ? c2 : c1;

        // Set Color
        if(!bit) continue;
        int mod = i % 3;
             if(mod == 0) red   += 1 << (7-i/3);
        else if(mod == 1) green += 1 << (7-i/3);
        else              blue  += 1 << (7-i/3);
    }

    // Colors are currently in [0, 255], scale to [0, 1]
    red /= 255.0f;
    green /= 255.0f;
    blue /= 255.0f;

    // Find the degree, since the leading coefficients might be zero
    int degree = -1;
    for (int i = ceil_terms - 1; i >= 0; i--) {
        if (coeffs[i].x != 0.0f || coeffs[i].y != 0.0f) {
            degree = i;
            break;
        }
    }
    if(degree < 1) return;

    cuFloatComplex roots[30];
    find_roots(coeffs, degree, roots);

    // Plot the roots
    for (int i = 0; i < degree; i++) {
        Cuda::vec2 point(cuCrealf(roots[i]), cuCimagf(roots[i]));
        Cuda::vec2 pixel = point_to_pixel_in_screen(point, lx_ty, rx_by, wh);
        d_gradient_circle(pixel.x, pixel.y, radius, red, green, blue, d_alpha, d_red, d_green, d_blue, wh, opacity);
    }
}

__device__ float sigmoid(float x) {
    return 3*x*x-2*x*x*x;
}

__global__ void finalize_color_kernel(unsigned int* d_pixels, float* d_alpha, float* d_red, float* d_green, float* d_blue,
        const Cuda::ivec2 wh, float brightness) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = wh.x * wh.y;

    if (idx >= total) return;

    float inv = 1.0f / (d_alpha[idx] + .000001);
    float r_mod = sigmoid(sigmoid(d_red  [idx] * inv));
    float g_mod = sigmoid(sigmoid(d_green[idx] * inv));
    float b_mod = sigmoid(sigmoid(d_blue [idx] * inv));

    brightness *= 256.0f;
    r_mod *= 256.0f-brightness;
    g_mod *= 256.0f-brightness;
    b_mod *= 256.0f-brightness;

    unsigned int a = Cuda::clamp(d_alpha[idx] * 2.5, 0.0f, 255.9f);
    unsigned int r = Cuda::clamp(r_mod + brightness, 0.0f, 255.9f);
    unsigned int g = Cuda::clamp(g_mod + brightness, 0.0f, 255.9f);
    unsigned int b = Cuda::clamp(b_mod + brightness, 0.0f, 255.9f);

    d_pixels[idx] = Cuda::argb(a, r, g, b);
}

extern "C" void draw_root_fractal(
    uint32_t* d_pixels,
    const Cuda::ivec2& wh,
    const std::complex<float>& c1,
    const std::complex<float>& c2,
    float terms,
    const Cuda::vec2& lx_ty,
    const Cuda::vec2& rx_by,
    float radius, float opacity, float brightness
) {
    int total = 1 << int(ceil(terms));

    float *d_alpha, *d_red, *d_green, *d_blue;
    size_t alloc_size = wh.x * wh.y * sizeof(float);
    cudaMalloc(&d_alpha, alloc_size);
    cudaMalloc(&d_red, alloc_size);
    cudaMalloc(&d_green, alloc_size);
    cudaMalloc(&d_blue, alloc_size);

    cuFloatComplex dc1 = make_cuFloatComplex(c1.real(), c1.imag());
    cuFloatComplex dc2 = make_cuFloatComplex(c2.real(), c2.imag());

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    root_fractal_kernel<<<blocks, threadsPerBlock>>>(d_alpha, d_red, d_green, d_blue, wh, dc1, dc2, terms, lx_ty, rx_by, radius, opacity);
    cudaDeviceSynchronize();

    int finalize_threadsPerBlock = 256;
    int finalize_blocks = (wh.x * wh.y + finalize_threadsPerBlock - 1) / finalize_threadsPerBlock;
    finalize_color_kernel<<<finalize_blocks, finalize_threadsPerBlock>>>(d_pixels, d_alpha, d_red, d_green, d_blue, wh, brightness);
    cudaDeviceSynchronize();

    cudaFree(d_alpha);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
}
