#include <cuda_runtime.h>
#include <cmath>
#include "../Host_Device_Shared/helpers.h"
#include "color.cuh"

__device__ float bicubic_weight(float t) {
    const float a = -0.5f;

    if (t < 0) t = -t;
    float t2 = t * t;
    float t3 = t2 * t;

    if (t <= 1.0f) {
        return (a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f;
    } else if (t < 2.0f) {
        return a * (t3 - 5.0f * t2 + 8.0f * t - 4.0f);
    } else {
        return 0.0f;
    }
};

__global__ void bicubic_scale_kernel(
    const uint32_t* in_pixels, const int in_w, const int in_h,
    uint32_t* out_pixels, const int out_w, const int out_h,
    const float x_ratio, const float y_ratio)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= out_w * out_h) return;

    int x = idx % out_w;
    int y = idx / out_w;

    float gx = x * x_ratio;
    float gy = y * y_ratio;

    int gxi = static_cast<int>(floorf(gx));
    int gyi = static_cast<int>(floorf(gy));

    float dx = gx - gxi;
    float dy = gy - gyi;

    float pa = 0.0f;
    float pr = 0.0f;
    float pg = 0.0f;
    float pb = 0.0f;

    // Iterate over the surrounding 4x4 block of pixels
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int xi = gxi + m;
            int yi = gyi + n;

            xi = Cuda::clamp(xi, 0, in_w - 1);
            yi = Cuda::clamp(yi, 0, in_h - 1);

            uint32_t pixel = in_pixels[yi * in_w + xi];
            float weight = bicubic_weight(dx - m) * bicubic_weight(dy - n);

            pa += weight * d_geta(pixel);
            pr += weight * d_getr(pixel);
            pg += weight * d_getg(pixel);
            pb += weight * d_getb(pixel);
        }
    }

    int ia = static_cast<int>(roundf(pa));
    int ir = static_cast<int>(roundf(pr));
    int ig = static_cast<int>(roundf(pg));
    int ib = static_cast<int>(roundf(pb));

    ia = min(255, max(0, ia));
    ir = min(255, max(0, ir));
    ig = min(255, max(0, ig));
    ib = min(255, max(0, ib));

    out_pixels[y * out_w + x] = d_argb(ia, ir, ig, ib);
}

extern "C" int cuda_bicubic_scale(const uint32_t* input_pixels, int input_w, int input_h, uint32_t* output_pixels, int output_w, int output_h) {
    uint32_t* d_input = nullptr;
    uint32_t* d_output = nullptr;

    size_t in_size = input_w * input_h * sizeof(uint32_t);
    size_t out_size = output_w * output_h * sizeof(uint32_t);

    cudaMalloc((void**)&d_input, in_size);
    cudaMemcpy(d_input, input_pixels, in_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_output, out_size);

    float x_ratio = static_cast<float>(input_w) / output_w;
    float y_ratio = static_cast<float>(input_h) / output_h;

    int numPixels = output_w * output_h;
    int blockSize = 256;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;
    bicubic_scale_kernel<<<numBlocks, blockSize>>>(
        d_input, input_w, input_h,
        d_output, output_w, output_h,
        x_ratio, y_ratio);
    cudaDeviceSynchronize();

    cudaMemcpy(output_pixels, d_output, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0; // success
}

__global__ void overlay_kernel(
    uint32_t* background, const Cuda::ivec2 b_wh,
    const uint32_t* foreground, const Cuda::ivec2 f_wh,
    const Cuda::vec2 center, const float opacity)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= f_wh.x * f_wh.y) return;

    Cuda::ivec2 f_pos(idx % f_wh.x, idx / f_wh.x);

    d_overlay_pixel(f_pos + floor(center), foreground[f_pos.y * f_wh.x + f_pos.x], opacity, background, b_wh);
}

__global__ void overlay_rotation_kernel(
    uint32_t* background, const Cuda::ivec2 b_wh,
    const uint32_t* foreground, const Cuda::ivec2 f_wh,
    const Cuda::vec2 center, const float opacity, const float angle_rad)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= b_wh.x * b_wh.y) return;

    Cuda::ivec2 b_pos(idx % b_wh.x, idx / b_wh.x);

    // Compute position relative to overlay top-left
    Cuda::vec2 rel_pos = b_pos - center;

    // Center of the foreground
    Cuda::vec2 fg_center = (f_wh - Cuda::ivec2(1, 1)) * 0.5f;

    // Translate to center, apply inverse rotation, translate back
    Cuda::vec2 f = rel_pos - fg_center;
    float cosA = cosf(angle_rad);
    float sinA = sinf(angle_rad);
    // inverse rotation by -angle -> use cos, -sin
    float srcx =  cosA * f.x + sinA * f.y + fg_center.x;
    float srcy = -sinA * f.x + cosA * f.y + fg_center.y;

    if (srcx < 0.0f || srcx >= static_cast<float>(f_wh.x - 1) ||
        srcy < 0.0f || srcy >= static_cast<float>(f_wh.y - 1)) {
        // Outside the source bounds or on boundary where bilinear needs neighbors
        return;
    }

    // Bilinear interpolation
    int x0 = static_cast<int>(floorf(srcx));
    int y0 = static_cast<int>(floorf(srcy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = srcx - x0;
    float sy = srcy - y0;

    uint32_t p00 = foreground[y0 * f_wh.x + x0];
    uint32_t p10 = foreground[y0 * f_wh.x + x1];
    uint32_t p01 = foreground[y1 * f_wh.x + x0];
    uint32_t p11 = foreground[y1 * f_wh.x + x1];

    float a00 = static_cast<float>(d_geta(p00));
    float r00 = static_cast<float>(d_getr(p00));
    float g00 = static_cast<float>(d_getg(p00));
    float b00 = static_cast<float>(d_getb(p00));

    float a10 = static_cast<float>(d_geta(p10));
    float r10 = static_cast<float>(d_getr(p10));
    float g10 = static_cast<float>(d_getg(p10));
    float b10 = static_cast<float>(d_getb(p10));

    float a01 = static_cast<float>(d_geta(p01));
    float r01 = static_cast<float>(d_getr(p01));
    float g01 = static_cast<float>(d_getg(p01));
    float b01 = static_cast<float>(d_getb(p01));

    float a11 = static_cast<float>(d_geta(p11));
    float r11 = static_cast<float>(d_getr(p11));
    float g11 = static_cast<float>(d_getg(p11));
    float b11 = static_cast<float>(d_getb(p11));

    // Interpolate along x
    float a0 = a00 * (1.0f - sx) + a10 * sx;
    float r0 = r00 * (1.0f - sx) + r10 * sx;
    float g0 = g00 * (1.0f - sx) + g10 * sx;
    float b0 = b00 * (1.0f - sx) + b10 * sx;

    float a1 = a01 * (1.0f - sx) + a11 * sx;
    float r1 = r01 * (1.0f - sx) + r11 * sx;
    float g1 = g01 * (1.0f - sx) + g11 * sx;
    float b1 = b01 * (1.0f - sx) + b11 * sx;

    // Interpolate along y
    float af = a0 * (1.0f - sy) + a1 * sy;
    float rf = r0 * (1.0f - sy) + r1 * sy;
    float gf = g0 * (1.0f - sy) + g1 * sy;
    float bf = b0 * (1.0f - sy) + b1 * sy;

    // Normalize fg alpha and apply global opacity
    float fg_alpha = (af / 255.0f) * opacity;
    if (fg_alpha <= 0.0f) return;

    d_overlay_pixel(b_pos, d_argb(255, rf, gf, bf), fg_alpha, background, b_wh);
}

extern "C" void cuda_overlay (
    uint32_t* background, const Cuda::ivec2& b_wh,
    const uint32_t* foreground, const Cuda::ivec2& f_wh,
    const Cuda::vec2& center, const float opacity, const float angle_rad)
{
    // Functionally equivalent to cuda_overlay, but the foreground is rotated about its center
    // by the specified angle (in radians) before being overlaid onto the background.
    if (opacity == 0.0f) return;
    float angle_mod = Cuda::extended_mod(angle_rad, 2.0f * M_PI);

    // TODO instead use the envelope surrounding the rotation INTERSECT the background itself
    int blockSize = 256;
    int numBlocks = (b_wh.x * b_wh.y + blockSize - 1) / blockSize;
    const float epsilon = 0.001f;
    if (angle_mod < epsilon || angle_mod > (2.0f * M_PI - epsilon)) {
        // If angle is effectively 0, skip rotation math and just do normal overlay
        overlay_kernel<<<numBlocks, blockSize>>>(
            background, b_wh,
            foreground, f_wh,
            center, opacity);
    } else {
        overlay_rotation_kernel<<<numBlocks, blockSize>>>(
            background, b_wh,
            foreground, f_wh,
            center, opacity, angle_rad);
    }
    cudaDeviceSynchronize();
}

extern "C" void cuda_zeroize_pixels(uint32_t* d_pixels, const Cuda::ivec2& wh) {
    cudaMemset(d_pixels, 0, wh.x * wh.y * sizeof(uint32_t));
}
