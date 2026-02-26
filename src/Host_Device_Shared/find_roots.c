#pragma once
// There's probably a better way to make this work both on CPU and GPU...

#define maxn 30
#ifdef __CUDACC__
__device__ void find_roots(const cuFloatComplex* coeffs_in, int degree, cuFloatComplex* roots) {
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

#else

#include <complex>

inline void check_answer(const std::complex<float>* coeffs, int degree, const std::complex<float>* roots) {
    // Multiply out (x - r1)(x - r2)...(x - rn) and compare to coeffs
    std::complex<float> prod[maxn + 1] = { std::complex<float>(0.0f, 0.0f) };
    prod[0] = std::complex<float>(1.0f, 0.0f); // Start with 1

    for (int i = 0; i < degree; ++i) {
        std::complex<float> r = roots[i];
        // Multiply current polynomial by (x - r)
        std::complex<float> new_prod[maxn + 1] = { std::complex<float>(0.0f, 0.0f) };
        for (int j = 0; j <= i; ++j) {
            new_prod[j + 1] += prod[j];          // x * prod[j]
            new_prod[j] += prod[j] * (-r);      // -r * prod[j]
        }
        for (int j = 0; j <= i + 1; ++j) {
            prod[j] = new_prod[j];
        }
    }

    // Now prod should be the coefficients of the polynomial with roots 'roots'
    // Compare to input coeffs
    float max_err = 0.0f;
    for (int i = 0; i <= degree; ++i) {
        float err = std::abs(prod[i] - coeffs[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err > 1e-2f) {
        throw std::runtime_error("Root finding failed: max error " + std::to_string(max_err) + " exceeds tolerance");
    }
}

inline void find_roots(const std::complex<float>* coeffs_in, int degree, std::complex<float>* roots) {
    // Use Durand-Kerner (Weierstrass) method to find all roots of polynomial
    // coeffs_in: coeffs[0..degree] where coeffs[i] corresponds to x^i
    // degree: n (degree of polynomial)
    if (degree <= 0) {
        return;
    }

    std::complex<float> coeffs[maxn + 1];
    for (int i = 0; i <= degree; ++i) {
        coeffs[i] = coeffs_in[i];
    }

    // Make monic: divide all coefficients by leading coefficient coeffs[degree]
    std::complex<float> leading = coeffs[degree];
    float leading_abs = std::abs(leading);
    if (leading_abs == 0.0f) {
        // Degenerate polynomial; just return zeros
        for (int i = 0; i < degree; ++i) roots[i] = std::complex<float>(0.0f, 0.0f);
        return;
    }
    for (int i = 0; i <= degree; ++i) {
        coeffs[i] = coeffs[i] / leading;
    }

    // Compute radius for initial guesses: 1 + max |a_i| for i=0..degree-1
    float max_coeff_abs = 0.0f;
    for (int i = 0; i < degree; ++i) {
        float aabs = std::abs(coeffs[i]);
        if (aabs > max_coeff_abs) max_coeff_abs = aabs;
    }
    float radius = 1.0f + max_coeff_abs;

    // Initialize roots on a circle
    const float PI2 = 6.28318530717958647692f;
    for (int i = 0; i < degree; ++i) {
        float angle = PI2 * i / degree;
        roots[i] = std::complex<float>(cos(angle)*radius, sin(angle)*radius);
    }

    const int max_iters = 1000;
    const float tol = 1e-6f;

    for (int iter = 0; iter < max_iters; ++iter) {
        float max_change = 0.0f;

        // For each root
        for (int i = 0; i < degree; ++i) {
            std::complex<float> xi = roots[i];

            // Evaluate polynomial p(xi) using Horner's method: coeffs[degree]*x^degree + ... + coeffs[0]
            std::complex<float> p = coeffs[degree];
            for (int k = degree - 1; k >= 0; --k) {
                p = p * xi + coeffs[k];
            }

            // Compute denominator: product_{j != i} (xi - xj)
            std::complex<float> denom = std::complex<float>(1.0f, 0.0f);
            for (int j = 0; j < degree; ++j) {
                if (j == i) continue;
                std::complex<float> diff = xi - roots[j];
                float diff_abs = std::abs(diff);
                if (diff_abs == 0.0f) {
                    // Perturb slightly to avoid zero division
                    diff = diff + std::complex<float>(1e-6f, 1e-6f);
                }
                denom *= diff;
            }

            float denom_abs = std::abs(denom);
            if (denom_abs == 0.0f) continue;

            std::complex<float> correction = p / denom;
            std::complex<float> new_xi = xi - correction;
            float change = std::abs(new_xi - xi);
            if (change > max_change) max_change = change;
            roots[i] = new_xi;
        }

        if (max_change < tol) break;
    }

    // Check answer by multiplying out (x - r1)(x - r2)...(x - rn) and comparing to monic coeffs
    check_answer(coeffs, degree, roots);
}

#endif
