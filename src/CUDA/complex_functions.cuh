#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../Host_Device_Shared/vec.h"

namespace cuCFunc{
    
// Complex to complex power (z ^ z)
__device__ cuComplex cuCpow(cuComplex base, cuComplex exponent) {
    float a = cuCrealf(base);
    float b = cuCimagf(base);
    float c = cuCrealf(exponent);
    float d = cuCimagf(exponent);
    if (a == 0.0 && b == 0.0)
        return make_cuComplex(0.0, 0.0);  // Zero raised to positive power is zero
    
    float r = sqrtf(a * a + b * b);  // Magnitude of the base
    float theta = atan2f(b, a);      // Argument of the base

    float new_r = powf(r, c) * expf(-d * theta);
    float new_theta = c * theta + d * logf(r);

    float new_cos, new_sin;
    sincosf(new_theta, &new_sin, &new_cos);

    return make_cuComplex(new_r * new_cos, new_r * new_sin);
}

// Complex to real power (z ^ n)
__device__ cuComplex cuCpow(cuComplex base, float exponent) {
    float a = cuCrealf(base);
    float b = cuCimagf(base);

    if (a == 0.0 && b == 0.0)
        return make_cuComplex(0.0, 0.0);  // Zero raised to positive power is zero
    
    float r = sqrtf(a * a + b * b);  // Magnitude of the base
    float theta = atan2f(b, a);      // Argument of the base

    float new_r = powf(r, exponent);
    float new_theta = exponent * theta;

    float new_cos, new_sin;
    sincosf(new_theta, &new_sin, &new_cos);

    return make_cuComplex(new_r * new_cos, new_r * new_sin);
}

//  Complex 3D to real power (z ^ n) (for mandelbulbs)
__device__ Cuda::vec3 cuCMpow(Cuda::vec3 base, float exponent) {
    float x = base.x;
    float y = base.y;
    float z = base.z;

    if (x == 0.0 && y == 0.0 && z == 0)
        return Cuda::vec3(0.0, 0.0, 0.0);  // Zero raised to positive power is zero
    
    float r = sqrtf(x * x + y * y + z * z);  // Magnitude of the base
    float theta = atan2f(z, x);              // Angle around z-axis
    float phi = asinf(y / r);                // Angle from north pole

    float new_r = powf(r, exponent);
    float new_theta = exponent * theta;
    float new_phi = exponent * phi;

    float new_cos_theta, new_sin_theta, new_cos_phi, new_sin_phi;
    sincosf(new_theta, &new_sin_theta, &new_cos_theta);
    sincosf(new_phi, &new_sin_phi, &new_cos_phi);

    return new_r * Cuda::vec3(new_cos_theta * new_cos_phi, -new_sin_phi, new_sin_theta * new_cos_phi);
}

// These functions take references to real and imaginary componenets separately to avoid conversion to and from cuComplex
// __forceinline__ is probably best so the compiler can (directly?) replace function calls with the underlying statements

// Complex to complex power (z ^ z)
__device__ __forceinline__ void cuCpow(float& zr, float& zi, const float xr, const float xi){
    if (zr == 0.0 && zi == 0.0){ // Zero raised to positive power is zero
        zr = 0;
        zi = 0;
        return;
    }
    float r = sqrtf(zr * zr + zi * zi);  // Magnitude of the base
    float theta = atan2f(zi, zr);        // Argument of the base

    float new_r = powf(r, xr) * expf(-xi * theta);
    float new_theta = xr * theta + xi * logf(r);

    float new_cos, new_sin;
    sincosf(new_theta, &new_sin, &new_cos);

    zr = new_r * new_cos;
    zi = new_r * new_sin;
}

// Complex to real power (z ^ n)
__device__ __forceinline__ void cuCpow(float& zr, float& zi, const float exponent){
    if (zr == 0.0 && zi == 0.0){ // Zero raised to positive power is zero
        zr = 0;
        zi = 0;
        return;
    }
    float r = sqrtf(zr * zr + zi * zi);  // Magnitude of the base
    float theta = atan2f(zi, zr);        // Argument of the base

    float new_r = powf(r, exponent);
    float new_theta = exponent * theta;

    float new_cos, new_sin;
    sincosf(new_theta, &new_sin, &new_cos);

    zr = new_r * new_cos;
    zi = new_r * new_sin;
}

__device__ __forceinline__ void squareZ(float& zr, float& zi){
    float zr_new = zr * zr - zi * zi; // Real part of z^2
    zi = 2.0 * zr * zi; // Imaginary part of z^2
    
    zr = zr_new;
}

__device__ __forceinline__ void cubeZ(float& zr, float& zi){
    float zr_new = zr * zr * zr - 3.0 * zr * zi * zi;  // Real part of z^3
    zi = 3.0 * zr * zr * zi - zi * zi * zi;  // Imaginary part of z^3

    zr = zr_new;
}

// Iterate z^x + c until bailout radius
__device__ int mandelbrot_iterations(
    cuComplex& z, const cuComplex& x, const cuComplex& c,
    int max_iterations, float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zr = cuCrealf(z);
    float zi = cuCimagf(z);
    const float xr = cuCrealf(x);
    const float xi = cuCimagf(x);
    const float cr = cuCrealf(c);
    const float ci = cuCimagf(c);

    for (; iterations < max_iterations; iterations++) {
        // Update z with z^x + c formula
        cuCpow(zr, zi, xr, xi);

        zr += cr;
        zi += ci;

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

// Iterate z^n + c until bailout radius (real exponent)
__device__ int mandelbrot_iterations(
    cuComplex& z, const float exponent, const cuComplex& c,
    int max_iterations, float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zr = cuCrealf(z);
    float zi = cuCimagf(z);
    const float cr = cuCrealf(c);
    const float ci = cuCimagf(c);

    for (; iterations < max_iterations; iterations++) {
        // Update z with z^n + c formula
        cuCpow(zr, zi, exponent);

        zr += cr;
        zi += ci;

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ int mandelbrot_iterations_2(
    cuComplex& z, const cuComplex& c,
    int max_iterations, float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0;

    // Extract real and imaginary parts of z and c
    float zr = cuCrealf(z);
    float zi = cuCimagf(z);
    const float cr = cuCrealf(c);
    const float ci = cuCimagf(c);

    for (; iterations < max_iterations; iterations++) {
        // Update z with z^2 + c formula
        squareZ(zr, zi);

        zr += cr;
        zi += ci;

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ int mandelbrot_iterations_3(
    cuComplex& z, const cuComplex& c,
    int max_iterations, float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0;

    // Extract real and imaginary parts of z and c
    float zr = cuCrealf(z);
    float zi = cuCimagf(z);
    const float cr = cuCrealf(c);
    const float ci = cuCimagf(c);

    for (; iterations < max_iterations; iterations++) {
        // Update z with z^3 + c formula
        cubeZ(zr, zi);

        zr += cr;
        zi += ci;

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

// Iterate z^n + c until bailout radius (real exponent)
__device__ int mandelbulb_iterations(
    const Cuda::vec3& z, const float exponent, const Cuda::vec3& c,
    int max_iterations, float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zx = z.x;
    float zy = z.y;
    float zz = z.z;
    const float cx = c.x;
    const float cy = c.y;
    const float cz = c.z;
    Cuda::vec3 z_new = z;

    for (; iterations < max_iterations; iterations++) {
        // Update z with z^n + c formula
        z_new = cuCMpow(z_new, exponent);

        z_new += c;

        sq_radius = z_new.x * z_new.x + z_new.y * z_new.y + z_new.z * z_new.z;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

}