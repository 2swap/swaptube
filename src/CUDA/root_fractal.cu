#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include "common_graphics.cuh"
#include "../Host_Device_Shared/find_roots.c"
#include "../Host_Device_Shared/helpers.h"

__device__ cuFloatComplex complex_pow(cuFloatComplex z, int n) {
    cuFloatComplex result = make_cuFloatComplex(1.0f, 0.0f);
    for (int i = 0; i < n; i++) {
        result = cuCmulf(result, z);
    }
    return result;
}

// Make a circular gradient on a pixel buffer
__device__ void d_gradient_circle(float cx, float cy, float radius, float red, float green, float blue, float* d_alpha, float* d_red, float* d_green, float* d_blue, int width, int height, float opa=1.0f) {
    // breakout if any part of the circle is outside of screen
    if (cx < 0 || cx >= width || cy < 0 || cy >= height)
        return;
    float radius2 = radius * radius;
    for (int x = cx - radius; x < cx + radius; x++) {
        float sdx = (x - cx)*(x - cx);
        for (int y = cy - radius; y < cy + radius; y++) {
            float sdy = (y - cy)*(y - cy);
            float dist2 = (sdx + sdy) / radius2;
            if (dist2 < 1.0f) {
                float final_opa = opa / (.03 + 240 * dist2 * dist2);
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    atomicAdd(&d_alpha[y * width + x],         final_opa);
                    atomicAdd(&d_red  [y * width + x], red   * final_opa);
                    atomicAdd(&d_green[y * width + x], green * final_opa);
                    atomicAdd(&d_blue [y * width + x], blue  * final_opa);
                }
            }
        }
    }
}

__global__ void root_fractal_kernel(float* d_alpha, float* d_red, float* d_green, float* d_blue, int w, int h, cuFloatComplex c1, cuFloatComplex c2, float terms, float lx, float ty, float rx, float by, float radius, float opacity, float rainbow) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ceil_terms = ceil(terms);
    unsigned int total = 1 << ceil_terms; // total number of polynomials

    if (idx >= total) return;

    cuFloatComplex coeffs[20];
    float red = 0.0f, green = 0.0f, blue = 0.0f;
    for (int i = 0; i < ceil_terms; i++) {
        // Set Coefficient
        bool bit = (idx >> i) & 1;
        coeffs[i] = bit ? c2 : c1;

        // Set Color
        if(!bit) continue;
        int mod = i % 3;
             if(mod == 0) red   += 1 << (7 - i/3);
        else if(mod == 1) green += 1 << (7 - i/3);
        else              blue  += 1 << (7 - i/3);
    }

    // Colors are currently in [0, 255], scale to [0, 1]
    red /= 255.0f;
    green /= 255.0f;
    blue /= 255.0f;
    //red   = lerp(0, red  , rainbow);
    //green = lerp(0, green, rainbow);
    //blue  = lerp(0, blue , rainbow);

    // Find the degree, since the leading coefficients might be zero
    int degree = -1;
    for (int i = ceil_terms - 1; i >= 0; i--) {
        if (coeffs[i].x != 0.0f || coeffs[i].y != 0.0f) {
            degree = i;
            break;
        }
    }
    if(degree < 1) return;

    cuFloatComplex roots[20];
    find_roots(coeffs, degree, roots);

    // Plot the roots
    for (int i = 0; i < degree; i++) {
        glm::vec2 point(cuCrealf(roots[i]), cuCimagf(roots[i]));
        glm::vec2 pixel = point_to_pixel(point, glm::vec2(lx, ty), glm::vec2(rx, by), glm::vec2(w, h));
        d_gradient_circle(pixel.x, pixel.y, radius, red, green, blue, d_alpha, d_red, d_green, d_blue, w, h, opacity);
    }
}

__global__ void finalize_color_kernel(unsigned int* d_pixels, float* d_alpha, float* d_red, float* d_green, float* d_blue, int w, int h) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = w * h;

    if (idx >= total) return;

    unsigned int a = clamp(d_alpha[idx]                                      , 0.0f, 255.9f);
    unsigned int r = clamp(d_red  [idx] * 256 / (d_alpha[idx] + .000001) + 64, 0.0f, 255.9f);
    unsigned int g = clamp(d_green[idx] * 256 / (d_alpha[idx] + .000001) + 64, 0.0f, 255.9f);
    unsigned int b = clamp(d_blue [idx] * 256 / (d_alpha[idx] + .000001) + 64, 0.0f, 255.9f);

    d_pixels[idx] = d_argb(a, r, g, b);
}

extern "C" void draw_root_fractal(
    unsigned int* pixels,
    int w,
    int h,
    std::complex<float> c1,
    std::complex<float> c2,
    float terms,
    float lx, float ty,
    float rx, float by,
    float radius, float opacity, float rainbow
) {
    int total = 1 << int(ceil(terms));

    float *d_alpha, *d_red, *d_green, *d_blue;
    cudaMalloc(&d_alpha, w * h * sizeof(float));
    cudaMalloc(&d_red, w * h * sizeof(float));
    cudaMalloc(&d_green, w * h * sizeof(float));
    cudaMalloc(&d_blue, w * h * sizeof(float));

    cuFloatComplex dc1 = make_cuFloatComplex(c1.real(), c1.imag());
    cuFloatComplex dc2 = make_cuFloatComplex(c2.real(), c2.imag());

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    root_fractal_kernel<<<blocks, threadsPerBlock>>>(d_alpha, d_red, d_green, d_blue, w, h, dc1, dc2, terms, lx, ty, rx, by, radius, opacity, rainbow);
    cudaDeviceSynchronize();

    unsigned int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(unsigned int));

    int finalize_threadsPerBlock = 256;
    int finalize_blocks = (w * h + finalize_threadsPerBlock - 1) / finalize_threadsPerBlock;
    finalize_color_kernel<<<finalize_blocks, finalize_threadsPerBlock>>>(d_pixels, d_alpha, d_red, d_green, d_blue, w, h);
    cudaDeviceSynchronize();

    cudaFree(d_alpha);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
