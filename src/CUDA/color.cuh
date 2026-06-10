#pragma once

#include <thrust/complex.h>
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/Color.h"

__device__ __forceinline__ void overlay_pixel(const Cuda::ivec2& pix, int col, float opacity, uint32_t* pixels, const Cuda::ivec2& wh) {
    if (pix.x < 0 || pix.x >= wh.x || pix.y < 0 || pix.y >= wh.y) return;
    opacity = Cuda::clamp(opacity, 0.0f, 1.0f);
    int idx = pix.y * wh.x + pix.x;
    int base = pixels[idx];
    int blended = Cuda::color_combine(base, col, opacity);
    pixels[idx] = blended;
}

__device__ __forceinline__ void atomic_overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, const Cuda::ivec2& wh) {
    if (x < 0 || x >= wh.x || y < 0 || y >= wh.y) return;
    opacity = Cuda::clamp(opacity, 0.0f, 1.0f);
    int idx = y * wh.x + x;

    unsigned int old_pixel = pixels[idx];
    int base = old_pixel;
    int blended = Cuda::color_combine(base, col, opacity);
    int new_pixel = blended;
    atomicCAS(&pixels[idx], old_pixel, new_pixel);
}

__device__ __forceinline__ int complex_to_srgb(const thrust::complex<float>& c, float ab_dilation, float dot_radius) {
    float mag = abs(c);
    if(mag < 1e-7) return Cuda::argb(255, 255, 255, 255); // white to dodge division by zero 
    thrust::complex<float> norm = (c * ab_dilation / mag + thrust::complex<float>(1,1)) * .5;
    float am = 2*atan(mag/dot_radius)/M_PI;
    return Cuda::OKLABtoRGB(255, 1-.8*am, Cuda::lerp(-.233888, .276216, norm.real()), Cuda::lerp(-.311528, .198570, norm.imag()));
}
