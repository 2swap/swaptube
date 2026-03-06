#include <cuda_runtime.h>

#define maxn 30

__device__ __forceinline__ void find_roots(const cuFloatComplex* coeffs_in, int degree, cuFloatComplex* roots) {
    // Use Durand-Kerner (Weierstrass) method to find all roots of polynomial
    // coeffs_in: coeffs[0..degree] where coeffs[i] corresponds to x^i
    // degree: n (degree of polynomial)
    if (degree <= 0) {
        return;
    }

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

    const int max_iters = 100;
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