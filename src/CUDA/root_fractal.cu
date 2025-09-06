#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdio>

__device__ cuFloatComplex complex_pow(cuFloatComplex z, int n) {
    cuFloatComplex result = make_cuFloatComplex(1.0f, 0.0f);
    for (int i = 0; i < n; i++) {
        result = cuCmulf(result, z);
    }
    return result;
}

// Evaluate polynomial p(z) = coeffs[0] + coeffs[1] z + ... + coeffs[n] z^n
__device__ cuFloatComplex eval_poly(const cuFloatComplex* coeffs, int n, cuFloatComplex z) {
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    for (int i = 0; i <= n; i++) {
        sum = cuCaddf(sum, cuCmulf(coeffs[i], complex_pow(z, i)));
    }
    return sum;
}

// Evaluate derivative of polynomial
__device__ cuFloatComplex eval_poly_derivative(const cuFloatComplex* coeffs, int n, cuFloatComplex z) {
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    for (int i = 1; i <= n; i++) {
        sum = cuCaddf(sum, make_cuFloatComplex(i * cuCrealf(coeffs[i]), i * cuCimagf(coeffs[i])));
        sum = cuCaddf(sum, cuCmulf(make_cuFloatComplex(i, 0.0f), complex_pow(z, i-1)));
    }
    return sum;
}

// Simple Newton-Raphson root finder for complex polynomials
__device__ cuFloatComplex find_root(const cuFloatComplex* coeffs, int n, cuFloatComplex z0, int max_iter=50, float tol=1e-4f) {
    cuFloatComplex z = z0;
    for (int i = 0; i < max_iter; i++) {
        cuFloatComplex f = eval_poly(coeffs, n, z);
        cuFloatComplex fprime = eval_poly_derivative(coeffs, n, z);
        float fprime_mag2 = cuCrealf(fprime)*cuCrealf(fprime) + cuCimagf(fprime)*cuCimagf(fprime);
        if (fprime_mag2 < 1e-8f) break; // avoid division by zero
        cuFloatComplex dz = cuCdivf(f, fprime);
        z = cuCsubf(z, dz);
        if (cuCabsf(dz) < tol) break;
    }
    return z;
}

__global__ void root_fractal_kernel(int* pixels, int w, int h, cuFloatComplex c1, cuFloatComplex c2, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = 1 << n; // total number of polynomials

    if (idx >= total) return;

    cuFloatComplex coeffs[20];
    for (int i = 0; i <= n; i++) {
        coeffs[i] = (idx & (1 << i)) ? c2 : c1;
    }

    // try roots starting from points on a circle
    for (int i = 0; i < n; i++) {
        float angle = 2.0f * 3.14159265359f * i / n;
        cuFloatComplex z0 = make_cuFloatComplex(cosf(angle), sinf(angle));
        cuFloatComplex root = find_root(coeffs, n, z0);

        // map root to pixel coordinates
        int px = (int)((root.x + 2.0f) / 4.0f * w);
        int py = (int)((root.y + 2.0f) / 4.0f * h);

        if (px >= 0 && px < w && py >= 0 && py < h) {
            int offset = 3 * (py * w + px);
            pixels[offset] = 0xffffffff;
        }
    }
}

extern "C" void draw_root_fractal(int* pixels, int w, int h, complex<float> c1, complex<float> c2, int n) {
    int total = 1 << n;
    int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(int));
    cudaMemcpy(d_pixels, pixels, w * h * sizeof(int), cudaMemcpyHostToDevice);

    cuFloatComplex dc1 = make_cuFloatComplex(c1.real(), c1.imag());
    cuFloatComplex dc2 = make_cuFloatComplex(c2.real(), c2.imag());

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    root_fractal_kernel<<<blocks, threadsPerBlock>>>(d_pixels, w, h, dc1, dc2, n);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

