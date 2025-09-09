#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdio>
#include "cuda_graphics.cu"

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

__device__ void find_roots(const cuFloatComplex* coeffs_in, int degree, cuFloatComplex* roots) {
    // Use Durand-Kerner (Weierstrass) method to find all roots of polynomial
    // coeffs_in: coeffs[0..degree] where coeffs[i] corresponds to x^i
    // degree: n (degree of polynomial)
    if (degree <= 0) {
        return;
    }

    const int maxn = 20;
    cuFloatComplex coeffs[maxn + 1];
    for (int i = 0; i <= degree; ++i) {
        coeffs[i] = coeffs_in[i];
    }

    // Make monic: divide all coefficients by leading coefficient coeffs[degree]
    cuFloatComplex leading = coeffs[degree];
    float leading_abs = cuCabsf(leading);
    if (leading_abs == 0.0f) {
        // Degenerate polynomial; just return zeros
        for (int i = 0; i < degree; ++i) roots[i] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }
    for (int i = 0; i <= degree; ++i) {
        coeffs[i] = cuCdivf(coeffs[i], leading);
    }

    // Compute radius for initial guesses: 1 + max |a_i| for i=0..degree-1
    float max_coeff_abs = 0.0f;
    for (int i = 0; i < degree; ++i) {
        float aabs = cuCabsf(coeffs[i]);
        if (aabs > max_coeff_abs) max_coeff_abs = aabs;
    }
    float radius = 1.0f + max_coeff_abs;

    // Initialize roots on a circle
    const float PI2 = 6.28318530717958647692f;
    for (int i = 0; i < degree; ++i) {
        float angle = PI2 * i / degree;
        roots[i] = make_cuFloatComplex(radius * cosf(angle), radius * sinf(angle));
    }

    const int max_iters = 200;
    const float tol = 1e-6f;

    for (int iter = 0; iter < max_iters; ++iter) {
        float max_change = 0.0f;

        // For each root
        for (int i = 0; i < degree; ++i) {
            cuFloatComplex xi = roots[i];

            // Evaluate polynomial p(xi) using Horner's method: coeffs[degree]*x^degree + ... + coeffs[0]
            cuFloatComplex p = coeffs[degree];
            for (int k = degree - 1; k >= 0; --k) {
                p = cuCaddf(cuCmulf(p, xi), coeffs[k]);
            }

            // Compute denominator: product_{j != i} (xi - xj)
            cuFloatComplex denom = make_cuFloatComplex(1.0f, 0.0f);
            for (int j = 0; j < degree; ++j) {
                if (j == i) continue;
                cuFloatComplex diff = cuCsubf(xi, roots[j]);
                float diff_abs = cuCabsf(diff);
                if (diff_abs == 0.0f) {
                    // Perturb slightly to avoid zero division
                    diff = cuCaddf(diff, make_cuFloatComplex(1e-6f, 1e-6f));
                }
                denom = cuCmulf(denom, diff);
            }

            float denom_abs = cuCabsf(denom);
            if (denom_abs == 0.0f) continue;

            cuFloatComplex correction = cuCdivf(p, denom);
            cuFloatComplex new_xi = cuCsubf(xi, correction);
            float change = cuCabsf(cuCsubf(new_xi, xi));
            if (change > max_change) max_change = change;
            roots[i] = new_xi;
        }

        if (max_change < tol) break;
    }
}

__global__ void root_fractal_kernel(unsigned int* pixels, int w, int h, cuFloatComplex c1, cuFloatComplex c2, int terms, float lx, float ty, float rx, float by) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = 1 << terms; // total number of polynomials
    int degree = terms - 1;

    if (idx >= total) return;

    cuFloatComplex coeffs[20];
    unsigned int color = 0xFF000000;
    for (int i = 0; i < terms; i++) {
        bool bit = (idx >> i) & 1;
        coeffs[i] = bit ? c2 : c1;
        if(!bit) continue;
        unsigned int color_or = 1 << 23;
        color_or >>= i/3;
        color_or >>= i%3 * 8;
        color |= color_or;
    }

    cuFloatComplex roots[20];
    find_roots(coeffs, degree, roots);

    // Plot the roots
    for (int i = 0; i < degree; i++) {
        glm::vec2 point(cuCrealf(roots[i]), cuCimagf(roots[i]));
        glm::vec2 pixel = point_to_pixel(point, glm::vec2(lx, ty), glm::vec2(rx, by), glm::vec2(w, h));
        int px = static_cast<int>(roundf(pixel.x));
        int py = static_cast<int>(roundf(pixel.y));
        if (px >= 0 && px < w && py >= 0 && py < h) {
            device_fill_circle(px, py, 1, color, pixels, w, h);
        }
    }
}

extern "C" void draw_root_fractal(
    unsigned int* pixels,
    int w,
    int h,
    complex<float> c1,
    complex<float> c2,
    int terms,
    float lx, float ty,
    float rx, float by
) {
    int total = 1 << terms;
    unsigned int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(unsigned int));
    cudaMemcpy(d_pixels, pixels, w * h * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cuFloatComplex dc1 = make_cuFloatComplex(c1.real(), c1.imag());
    cuFloatComplex dc2 = make_cuFloatComplex(c2.real(), c2.imag());

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    root_fractal_kernel<<<blocks, threadsPerBlock>>>(d_pixels, w, h, dc1, dc2, terms, lx, ty, rx, by);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
