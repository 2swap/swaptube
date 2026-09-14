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

// Sample in_pixels at the source-texel coordinate (gx, gy) with a Catmull-Rom
// bicubic kernel whose support is stretched by (support_x, support_y). Pass
// support == 1 for plain interpolation (magnify / 1:1); pass the minification
// factor (source texels spanned per output texel) when downscaling so the
// kernel also low-pass filters and the result doesn't alias. Returns the
// weight-normalized channels as floats in [0, 255], packed as (a, r, g, b).
__device__ Cuda::vec4 sample_bicubic(
    const uint32_t* in_pixels, const Cuda::ivec2 in_wh,
    const float gx, const float gy,
    const float support_x, const float support_y)
{
    const int gxi = static_cast<int>(floorf(gx));
    const int gyi = static_cast<int>(floorf(gy));
    const float dx = gx - gxi;
    const float dy = gy - gyi;

    const int rx = static_cast<int>(ceilf(2.0f * support_x));
    const int ry = static_cast<int>(ceilf(2.0f * support_y));

    float pa = 0.0f, pr = 0.0f, pg = 0.0f, pb = 0.0f, wsum = 0.0f;
    for (int n = -ry; n <= ry + 1; n++) {
        const float wy = bicubic_weight((n - dy) / support_y);
        const int yi = Cuda::clamp(gyi + n, 0, in_wh.y - 1);
        for (int m = -rx; m <= rx + 1; m++) {
            const float wx = bicubic_weight((m - dx) / support_x);
            const int xi = Cuda::clamp(gxi + m, 0, in_wh.x - 1);

            const uint32_t p = in_pixels[yi * in_wh.x + xi];
            const float w = wx * wy;
            pa += w * Cuda::geta(p);
            pr += w * Cuda::getr(p);
            pg += w * Cuda::getg(p);
            pb += w * Cuda::getb(p);
            wsum += w;
        }
    }

    const float inv = wsum > 0.0f ? 1.0f / wsum : 0.0f;
    return Cuda::vec4(pa * inv, pr * inv, pg * inv, pb * inv);
}

__device__ __forceinline__ uint32_t pack_argb_saturate(const Cuda::vec4& c) {
    return Cuda::argb(
        min(255, max(0, static_cast<int>(roundf(c.x)))),
        min(255, max(0, static_cast<int>(roundf(c.y)))),
        min(255, max(0, static_cast<int>(roundf(c.z)))),
        min(255, max(0, static_cast<int>(roundf(c.w)))));
}

// Crop the normalized [crop_tl, crop_br] sub-rectangle of the source, resample
// it (anti-aliased bicubic) to fill out_wh, and multiply RGB by darken_factor.
// With crop_tl=(0,0), crop_br=(1,1), darken_factor=1 this is a plain resize.
__global__ void crop_scale_darken_kernel(
    const uint32_t* in_pixels, const Cuda::ivec2 in_wh,
    uint32_t* out_pixels, const Cuda::ivec2 out_wh,
    const Cuda::vec2 crop_tl, const Cuda::vec2 crop_br,
    const float darken_factor)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= out_wh.x * out_wh.y) return;

    const int x = idx % out_wh.x;
    const int y = idx / out_wh.x;

    const float u = (x + 0.5f) / out_wh.x;
    const float v = (y + 0.5f) / out_wh.y;

    // Output texel center -> source texel space.
    const float gx = (crop_tl.x + u * (crop_br.x - crop_tl.x)) * in_wh.x - 0.5f;
    const float gy = (crop_tl.y + v * (crop_br.y - crop_tl.y)) * in_wh.y - 0.5f;

    // Source texels spanned per output texel; > 1 only when minifying.
    const float support_x = fmaxf(1.0f, (crop_br.x - crop_tl.x) * in_wh.x / out_wh.x);
    const float support_y = fmaxf(1.0f, (crop_br.y - crop_tl.y) * in_wh.y / out_wh.y);

    Cuda::vec4 c = sample_bicubic(in_pixels, in_wh, gx, gy, support_x, support_y);
    c.y *= darken_factor;
    c.z *= darken_factor;
    c.w *= darken_factor;

    out_pixels[y * out_wh.x + x] = pack_argb_saturate(c);
}

extern "C" void cuda_crop_scale_darken_device(
    const uint32_t* d_input, const Cuda::ivec2& in_wh,
    uint32_t* d_output, const Cuda::ivec2& out_wh,
    const Cuda::vec2& crop_tl, const Cuda::vec2& crop_br,
    const float darken_factor)
{
    const int numPixels = out_wh.x * out_wh.y;
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;
    crop_scale_darken_kernel<<<numBlocks, blockSize>>>(
        d_input, in_wh, d_output, out_wh, crop_tl, crop_br, darken_factor);
    cudaDeviceSynchronize();
}

// Plain host-to-host resize: a full-frame crop with no darkening. Handles the
// device round trip for callers that hold their pixels on the host.
extern "C" int cuda_bicubic_scale(const uint32_t* input_pixels, int input_w, int input_h, uint32_t* output_pixels, int output_w, int output_h) {
    uint32_t* d_input = nullptr;
    uint32_t* d_output = nullptr;

    const size_t in_size = static_cast<size_t>(input_w) * input_h * sizeof(uint32_t);
    const size_t out_size = static_cast<size_t>(output_w) * output_h * sizeof(uint32_t);

    cudaMalloc((void**)&d_input, in_size);
    cudaMemcpy(d_input, input_pixels, in_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output, out_size);

    cuda_crop_scale_darken_device(
        d_input, Cuda::ivec2(input_w, input_h),
        d_output, Cuda::ivec2(output_w, output_h),
        Cuda::vec2(0.0f, 0.0f), Cuda::vec2(1.0f, 1.0f), 1.0f);

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
    Cuda::ivec2 b_pos(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    if (b_pos.x >= b_wh.x || b_pos.y >= b_wh.y) return;

    Cuda::ivec2 top_left = floor(center - (f_wh * 0.5f));
    Cuda::ivec2 f_pos(b_pos - top_left);
    if (f_pos.x < 0 || f_pos.x >= f_wh.x || f_pos.y < 0 || f_pos.y >= f_wh.y) return;

    overlay_pixel(b_pos, foreground[f_pos.y * f_wh.x + f_pos.x], opacity, background, b_wh);
}

__global__ void overlay_rotation_kernel(
    uint32_t* background, const Cuda::ivec2 b_wh,
    const uint32_t* foreground, const Cuda::ivec2 f_wh,
    const Cuda::vec2 center, const float opacity, const float angle_rad)
{
    Cuda::ivec2 b_pos(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    if (b_pos.x >= b_wh.x || b_pos.y >= b_wh.y) return;

    // Position relative to the overlay center - zero exactly where the foreground's
    // own center should be sampled from, so this is already the vector to rotate.
    Cuda::vec2 rel_pos = b_pos - center;

    // Center of the foreground
    Cuda::vec2 fg_center = (f_wh - Cuda::ivec2(1, 1)) * 0.5f;

    // Apply inverse rotation, then re-express relative to the foreground's top-left corner.
    float cosA = cosf(angle_rad);
    float sinA = sinf(angle_rad);
    // inverse rotation by -angle -> use cos, -sin
    float srcx =  cosA * rel_pos.x + sinA * rel_pos.y + fg_center.x;
    float srcy = -sinA * rel_pos.x + cosA * rel_pos.y + fg_center.y;

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

    float a00 = static_cast<float>(Cuda::geta(p00));
    float r00 = static_cast<float>(Cuda::getr(p00));
    float g00 = static_cast<float>(Cuda::getg(p00));
    float b00 = static_cast<float>(Cuda::getb(p00));

    float a10 = static_cast<float>(Cuda::geta(p10));
    float r10 = static_cast<float>(Cuda::getr(p10));
    float g10 = static_cast<float>(Cuda::getg(p10));
    float b10 = static_cast<float>(Cuda::getb(p10));

    float a01 = static_cast<float>(Cuda::geta(p01));
    float r01 = static_cast<float>(Cuda::getr(p01));
    float g01 = static_cast<float>(Cuda::getg(p01));
    float b01 = static_cast<float>(Cuda::getb(p01));

    float a11 = static_cast<float>(Cuda::geta(p11));
    float r11 = static_cast<float>(Cuda::getr(p11));
    float g11 = static_cast<float>(Cuda::getg(p11));
    float b11 = static_cast<float>(Cuda::getb(p11));

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

    overlay_pixel(b_pos, Cuda::argb(255, rf, gf, bf), fg_alpha, background, b_wh);
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
    dim3 blockSize(16, 16);
    dim3 numBlocks((b_wh.x + blockSize.x - 1) / blockSize.x, (b_wh.y + blockSize.y - 1) / blockSize.y);
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
