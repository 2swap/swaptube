#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdio>
#include "cuda_graphics.cu"
#include "../Host_Device_Shared/find_roots.c"

__device__ glm::vec2 point_to_pixel(const glm::vec2& point, const glm::vec2& lx_ty, const glm::vec2& rx_by, const glm::vec2& wh) {
    const glm::vec2 flip = (point - lx_ty) * wh / (rx_by - lx_ty);
    return glm::vec2(flip.x, wh.y-1-flip.y);
}

__device__ cuFloatComplex complex_pow(cuFloatComplex z, int n) {
    cuFloatComplex result = make_cuFloatComplex(1.0f, 0.0f);
    for (int i = 0; i < n; i++) {
        result = cuCmulf(result, z);
    }
    return result;
}

__global__ void root_fractal_kernel(unsigned int* pixels, int w, int h, cuFloatComplex c1, cuFloatComplex c2, float terms, float lx, float ty, float rx, float by, float radius, float opacity, float rainbow) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ceil_terms = ceil(terms);
    unsigned int floor_terms = floor(terms);
    unsigned int total = 1 << ceil_terms; // total number of polynomials

    if (idx >= total) return;
    if (idx >= total/2 && floor_terms != ceil_terms) {
        // For transitional term count, fade in/out newly introduced polynomials
        float one_minus_frac = 1 - (terms - floor_terms);
        float opacity_multiplier = 1-one_minus_frac*one_minus_frac*one_minus_frac;
        opacity *= opacity_multiplier;
    }

    cuFloatComplex coeffs[20];
    unsigned int color = 0xFF3f3f3f;
    for (int i = 0; i < ceil_terms; i++) {
        bool bit = (idx >> i) & 1;
        coeffs[i] = bit ? c2 : c1;
        if(!bit) continue;
        unsigned int color_or = 1 << 23;
        color_or >>= i/3;
        color_or >>= i%3 * 8;
        color |= color_or;
    }

    color = device_colorlerp(0xFFFFFFFF, color, rainbow);

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
        device_gradient_circle(pixel.x, pixel.y, radius, color, pixels, w, h, opacity);
    }
}

extern "C" void draw_root_fractal(
    unsigned int* pixels,
    int w,
    int h,
    complex<float> c1,
    complex<float> c2,
    float terms,
    float lx, float ty,
    float rx, float by,
    float radius, float opacity, float rainbow
) {
    int total = 1 << int(ceil(terms));
    unsigned int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(unsigned int));
    cudaMemcpy(d_pixels, pixels, w * h * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cuFloatComplex dc1 = make_cuFloatComplex(c1.real(), c1.imag());
    cuFloatComplex dc2 = make_cuFloatComplex(c2.real(), c2.imag());

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    root_fractal_kernel<<<blocks, threadsPerBlock>>>(d_pixels, w, h, dc1, dc2, terms, lx, ty, rx, by, radius, opacity, rainbow);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
