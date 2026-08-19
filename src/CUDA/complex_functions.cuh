#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../Host_Device_Shared/vec.h"

namespace {

// General prefix naming patterns:
// cuC : cuComplex
// c : separate variable complex
// m3 : mandelbulb triplex numbers
// q : quaternion (unfinished))

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
__device__ Cuda::vec3 m3pow(Cuda::vec3& base, float exponent) {
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

    new_cos_phi *= new_r;

    return Cuda::vec3(new_cos_theta * new_cos_phi, new_r * -new_sin_phi, new_sin_theta * new_cos_phi);
}

// These functions take references to real and imaginary componenets separately to avoid conversion to and from cuComplex
// __forceinline__ is probably best so the compiler can (directly?) replace function calls with the underlying statements

__device__ __forceinline__ void cmult(float& zr, float& zi, const float ar, const float ai){
    float zr_new = zr * ar - zi * ai; // Real part of z * a
    zi = zr * ai + zi * ar; // Imaginary part of z * a

    zr = zr_new;
}

// Complex to complex power (z ^ z)
__device__ __forceinline__ void cpow(float& zr, float& zi, const float xr, const float xi){
    if (zr == 0.0 && zi == 0.0) // Zero raised to positive power is zero
        return;
    
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
__device__ __forceinline__ void cpow(float& zr, float& zi, const float exponent){
    if (zr == 0.0 && zi == 0.0) // Zero raised to positive power is zero
        return;
    
    float r = sqrtf(zr * zr + zi * zi);  // Magnitude of the base
    float theta = atan2f(zi, zr);        // Argument of the base

    float new_r = powf(r, exponent);
    float new_theta = exponent * theta;

    float new_cos, new_sin;
    sincosf(new_theta, &new_sin, &new_cos);

    zr = new_r * new_cos;
    zi = new_r * new_sin;
}

// Simple complex polynomial a1 * z ^ x1 + a2 * z ^ x2 + a3 * z ^ x3 + a4 * z ^ x4 + c, for real a and x
__device__ __forceinline__ void cPoly(float& zr, float& zi, 
    const float a1, const float x1, 
    const float a2, const float x2, 
    const float a3, const float x3, 
    const float a4, const float x4,
    const float cr, const float ci,
    const char burning = 0, const char conj = 0
){
    float zr1 = (burning & 0b10000000) >> 7 ? fabsf(zr) : zr, zi1 = (burning & 0b01000000) >> 6 ? fabsf(zi) : zi;
    if(conj & 0b10000000){
        zr1 = -zr1;
    }
    if(conj & 0b01000000){
        zi1 = -zi1;
    }
    cpow(zr1, zi1, x1);
    zr1 *= a1;
    zi1 *= a1;

    float zr2 = (burning & 0b00100000) >> 5 ? fabsf(zr) : zr, zi2 = (burning & 0b00010000) >> 4 ? fabsf(zi) : zi;
    if(conj & 0b00100000){
        zr2 = -zr2;
    }
    if(conj & 0b00010000){
        zi2 = -zi2;
    }
    cpow(zr2, zi2, x2);
    zr2 *= a2;
    zi2 *= a2;

    float zr3 = (burning & 0b00001000) >> 3 ? fabsf(zr) : zr, zi3 = (burning & 0b00000100) >> 2 ? fabsf(zi) : zi;
    if(conj & 0b00001000){
        zr3 = -zr3;
    }
    if(conj & 0b00000100){
        zi3 = -zi3;
    }
    cpow(zr3, zi3, x3);
    zr3 *= a3;
    zi3 *= a3;

    float zr4 = (burning & 0b00000010) >> 1 ? fabsf(zr) : zr, zi4 = (burning & 0b00000001) ? fabsf(zi) : zi;
    if(conj & 0b00000010){
        zr4 = -zr4;
    }
    if(conj & 0b00000001){
        zi4 = -zi4;
    }
    cpow(zr4, zi4, x4);
    zr4 *= a4;
    zi4 *= a4;

    zr = zr1 + zr2 + zr3 + zr4 + cr;
    zi = zi1 + zi2 + zi3 + zi4 + ci;    
}

// Complex complex polynomial a1 * z ^ x1 + a2 * z ^ x2 + a3 * z ^ x3 + a4 * z ^ x4 + c, for complex a and x
__device__ __forceinline__ void cPoly(float& zr, float& zi, 
    const float a1r, const float a1i, const float x1r, const float x1i, 
    const float a2r, const float a2i, const float x2r, const float x2i, 
    const float a3r, const float a3i, const float x3r, const float x3i, 
    const float a4r, const float a4i, const float x4r, const float x4i,
    const float cr, const float ci,
    const char burning = 0, const char conj = 0
){
    float zr1 = (burning & 0b10000000) >> 7 ? fabsf(zr) : zr, zi1 = (burning & 0b01000000) >> 6 ? fabsf(zi) : zi;
    if(conj & 0b10000000){
        zr1 = -zr1;
    }
    if(conj & 0b01000000){
        zi1 = -zi1;
    }
    cpow(zr1, zi1, x1r, x1i);
    cmult(zr1, zi1, a1r, a1i);

    float zr2 = (burning & 0b00100000) >> 5 ? fabsf(zr) : zr, zi2 = (burning & 0b00010000) >> 4 ? fabsf(zi) : zi;
    if(conj & 0b00100000){
        zr2 = -zr2;
    }
    if(conj & 0b00010000){
        zi2 = -zi2;
    }
    cpow(zr2, zi2, x2r, x2i);
    cmult(zr2, zi2, a2r, a2i);

    float zr3 = (burning & 0b00001000) >> 3 ? fabsf(zr) : zr, zi3 = (burning & 0b00000100) >> 2 ? fabsf(zi) : zi;
    if(conj & 0b00001000){
        zr3 = -zr3;
    }
    if(conj & 0b00000100){
        zi3 = -zi3;
    }
    cpow(zr3, zi3, x3r, x3i);
    cmult(zr3, zi3, a3r, a3i);

    float zr4 = (burning & 0b00000010) >> 1 ? fabsf(zr) : zr, zi4 = (burning & 0b00000001) ? fabsf(zi) : zi;
    if(conj & 0b00000010){
        zr4 = -zr4;
    }
    if(conj & 0b00000001){
        zi4 = -zi4;
    }
    cpow(zr4, zi4, x4r, x4i);
    cmult(zr4, zi4, a4r, a4i);

    zr = zr1 + zr2 + zr3 + zr4 + cr;
    zi = zi1 + zi2 + zi3 + zi4 + ci;
}

// Complex complex polynomial with c (a1 + ac1 * c) * z ^ x1 + (a2 + ac2 * c) * z ^ x2 + (a3 + ac3 * c) * z ^ x3 + (a4 + ac4 * c) * z ^ x4 + c, for complex a, ac, and x
__device__ __forceinline__ void cPolyC(float& zr, float& zi, 
    const float a1r, const float a1i, const float ac1r, const float ac1i, const float x1r, const float x1i, 
    const float a2r, const float a2i, const float ac2r, const float ac2i, const float x2r, const float x2i, 
    const float a3r, const float a3i, const float ac3r, const float ac3i, const float x3r, const float x3i, 
    const float a4r, const float a4i, const float ac4r, const float ac4i, const float x4r, const float x4i, 
    const float cr, const float ci,
    const char burning = 0, const char conj = 0
){
    float zr1 = (burning & 0b10000000) >> 7 ? fabsf(zr) : zr, zi1 = (burning & 0b01000000) >> 6 ? fabsf(zi) : zi;
    if(conj & 0b10000000){
        zr1 = -zr1;
    }
    if(conj & 0b01000000){
        zi1 = -zi1;
    }
    float ac1r_new = ac1r, ac1i_new = ac1i;
    cmult(ac1r_new, ac1i_new, cr, ci);
    cpow(zr1, zi1, x1r, x1i);
    cmult(zr1, zi1, a1r + ac1r_new, a1i + ac1i_new);

    float zr2 = (burning & 0b00100000) >> 4 ? fabsf(zr) : zr, zi2 = (burning & 0b00010000) >> 4 ? fabsf(zi) : zi;
    if(conj & 0b00100000){
        zr2 = -zr2;
    }
    if(conj & 0b00010000){
        zi2 = -zi2;
    }
    float ac2r_new = ac2r, ac2i_new = ac2i;
    cmult(ac2r_new, ac2i_new, cr, ci);
    cpow(zr2, zi2, x2r, x2i);
    cmult(zr2, zi2, a2r + ac2r_new, a2i + ac2i_new);

    float zr3 = (burning & 0b00001000) >> 3 ? fabsf(zr) : zr, zi3 = (burning & 0b00000100) >> 2 ? fabsf(zi) : zi;
    if(conj & 0b00001000){
        zr3 = -zr3;
    }
    if(conj & 0b00000100){
        zi3 = -zi3;
    }
    float ac3r_new = ac3r, ac3i_new = ac3i;
    cmult(ac3r_new, ac3i_new, cr, ci);
    cpow(zr3, zi3, x3r, x3i);
    cmult(zr3, zi3, a3r + ac3r_new, a3i + ac3i_new);

    float zr4 = (burning & 0b00000010) >> 1 ? fabsf(zr) : zr, zi4 = (burning & 0b00000001) ? fabsf(zi) : zi;
    if(conj & 0b00000010){
        zr4 = -zr4;
    }
    if(conj & 0b00000001){
        zi4 = -zi4;
    }
    float ac4r_new = ac4r, ac4i_new = ac4i;
    cmult(ac4r_new, ac4i_new, cr, ci);
    cpow(zr4, zi4, x4r, x4i);
    cmult(zr4, zi4, a4r + ac4r_new, a4i + ac4i_new);

    zr = zr1 + zr2 + zr3 + zr4 + cr;
    zi = zi1 + zi2 + zi3 + zi4 + ci;
}

//  Complex 3D to real power (z ^ n) (for mandelbulbs)
__device__ __forceinline__ void m3pow(float& zx, float& zy, float& zz, float exponent) {
    if (zx == 0.0 && zy == 0.0 && zz == 0.0) // Zero raised to positive power is zero
        return;  

    float r = sqrtf(zx * zx + zy * zy + zz * zz);  // Magnitude of the base
    float theta = atan2f(zz, zx);              // Angle around z-axis
    float phi = asinf(zy / r);                // Angle from north pole

    float new_r = powf(r, exponent);
    float new_theta = exponent * theta;
    float new_phi = exponent * phi;

    float new_cos_theta, new_sin_theta, new_cos_phi, new_sin_phi;
    sincosf(new_theta, &new_sin_theta, &new_cos_theta);
    sincosf(new_phi, &new_sin_phi, &new_cos_phi);

    float scaled_cos_phi = new_cos_phi * new_r;

    zx = new_cos_theta * scaled_cos_phi;
    zy = new_r * -new_sin_phi;
    zz = new_sin_theta * scaled_cos_phi;
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

// Iterate complex polynomial until bailout radius
__device__ int mandelRealPoly_iterations(
    const float zr0, const float zi0,
    const float a1, const float x1, 
    const float a2, const float x2, 
    const float a3, const float x3, 
    const float a4, const float x4, 
    const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;

    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Update z with polynomial formula
        cPoly(zr, zi, a1, x1, a2, x2, a3, x3, a4, x4, cr, ci, burning, conj);

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

// Iterate complex polynomial with complex coefficients and exponents until bailout radius
__device__ int mandelPoly_iterations(
    const float zr0, const float zi0,
    const float a1r, const float a1i, const float x1r, const float x1i, 
    const float a2r, const float a2i, const float x2r, const float x2i, 
    const float a3r, const float a3i, const float x3r, const float x3i, 
    const float a4r, const float a4i, const float x4r, const float x4i,
    const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Update z with polynomial formula
        cPoly(zr, zi, a1r, a1i, x1r, x1i, a2r, a2i, x2r, x2i, a3r, a3i, x3r, x3i, a4r, a4i, x4r, x4i, cr, ci, burning, conj);

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ int mandelPolyC_iterations(
    const float zr0, const float zi0,
    const float a1r, const float a1i, const float ac1r, const float ac1i, const float x1r, const float x1i, 
    const float a2r, const float a2i, const float ac2r, const float ac2i, const float x2r, const float x2i, 
    const float a3r, const float a3i, const float ac3r, const float ac3i, const float x3r, const float x3i, 
    const float a4r, const float a4i, const float ac4r, const float ac4i, const float x4r, const float x4i,
    const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Update z with polynomial formula
        cPolyC(zr, zi, a1r, a1i, ac1r, ac1i, x1r, x1i, a2r, a2i, ac2r, ac2i, x2r, x2i, a3r, a3i, ac3r, ac3i, x3r, x3i, a4r, a4i, ac4r, ac4i, x4r, x4i, cr, ci, burning, conj);

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ __forceinline__ float smooth_iterations(float iters, float power, float sq_radius, float bailout_radius_sq) {
    return iters - (logf(logf(sq_radius) / logf(bailout_radius_sq)) / logf(power));
}

__device__ float potential(float sq_radius, float power, int iterations){
    return 0.5 * logf(sq_radius) / powf(power, iterations);
}

__device__ __forceinline__ void boettcher(float& zr, float& zi, float k, int n){
    float p = powf(k, -n);
    cpow(zr, zi, p);
}

// Iterate z^x + c until bailout radius
__device__ int mandelbrot_iterations(
    const float zr0, const float zi0, const float xr, const float xi, const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Absolute values for burning ship stuff
        zr = (burning & 0b10) >> 1 ? fabsf(zr) : zr;
        zi = (burning & 0b01) ? fabsf(zi) : zi;
        if(conj & 0b10){
            zr = -zr;
        }
        if(conj & 0b01){
            zi = -zi;
        }

        // Update z with z^x + c formula
        cpow(zr, zi, xr, xi);

        zr += cr;
        zi += ci;

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

// Iterate z^n + c until bailout radius (real exponent)
__device__ int mandelbrot_iterations(
    const float zr0, const float zi0, const float exponent, const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;
    
    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Absolute values for burning ship stuff
        zr = (burning & 0b10) >> 1 ? fabsf(zr) : zr;
        zi = (burning & 0b01) ? fabsf(zi) : zi;
        if(conj & 0b10){
            zr = -zr;
        }
        if(conj & 0b01){
            zi = -zi;
        }

        // Update z with z^n + c formula
        cpow(zr, zi, exponent);

        zr += cr;
        zi += ci;

        sq_radius = zr * zr + zi * zi;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ int mandelbrot_iterations_2(
    const float zr0, const float zi0, const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;

    // Extract real and imaginary parts of z and c
    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Absolute values for burning ship stuff
        zr = (burning & 0b10) >> 1 ? fabsf(zr) : zr;
        zi = (burning & 0b01) ? fabsf(zi) : zi;
        if(conj & 0b10){
            zr = -zr;
        }
        if(conj & 0b01){
            zi = -zi;
        }

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
    const float zr0, const float zi0, const float cr, const float ci,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius,
    const char burning = 0, const char conj = 0
) {
    int iterations = 0;
    sq_radius = 0;

    // Extract real and imaginary parts of z and c
    float zr = zr0;
    float zi = zi0;

    for (; iterations < max_iterations; iterations++) {
        // Absolute values for burning ship stuff
        zr = burning & 0b10 >> 1 ? fabsf(zr) : zr;
        zi = burning & 0b01 ? fabsf(zi) : zi;
        if(conj & 0b10){
            zr = -zr;
        }
        if(conj & 0b01){
            zi = -zi;
        }

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
    const int max_iterations, const float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0.0f;
    
    float zx = z.x;
    float zy = z.y;
    float zz = z.z;
    const float cx = c.x;
    const float cy = c.y;
    const float cz = c.z;

    for (; iterations < max_iterations; iterations++) {
        zx=(zx);
        zy=(zy);
        zz=(zz);

        // Update z with z^n + c formula
        m3pow(zx, zy, zz, exponent);

        zx += cx;
        zy += cy;
        zz += cz;

        sq_radius = zx * zx + zy * zy + zz * zz;

        if (sq_radius > bailout_radius_sq) return iterations;
    }
    
    return max_iterations; // No bailout, maximum iterations reached
}

__device__ Cuda::vec3 jacobiMult(Cuda::vec3& a, Cuda::vec3& b){
    return Cuda::vec3(
        a.x * b.x - a.y * b.z + a.z * b.y,
        a.x * b.y + a.y * b.x - a.z * b.z,
        a.x * b.z + a.y * b.y + a.z * b.x
    );
}

__device__ int jacobibrot2_iterations(
    const Cuda::vec3& z, const Cuda::vec3& c,
    const int max_iterations, const float bailout_radius_sq, float& sq_radius
) {
    int iterations = 0;
    sq_radius = 0;
    
    Cuda::vec3 current_z = z;

    for(; iterations < max_iterations; iterations++){
        current_z = jacobiMult(current_z, current_z);
        current_z += c;

        sq_radius = current_z.x * current_z.x + current_z.y * current_z.y + current_z.z * current_z.z;

        if(sq_radius > bailout_radius_sq) return iterations;
    }

    return max_iterations; // No bailout, maximum iterations reached
}

}