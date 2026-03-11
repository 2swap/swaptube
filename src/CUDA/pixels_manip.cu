#include <cuda_runtime.h>
#include <cmath>
#include "color.cuh" // Contains implementation of argb, geta, etc

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
    const unsigned int* in_pixels, const Cuda::vec2 in_size,
    unsigned int* out_pixels, const Cuda::vec2 out_size,
    const Cuda::vec2 ratio)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= out_size.x * out_size.y) return;

    Cuda::vec2 xy(idx % (int)out_size.x, idx / out_size.x);

    Cuda::vec2 g(xy * ratio);

    Cuda::vec2 gi(floorf(g.x), floorf(g.y));

    Cuda::vec2 delta(g - gi);

    float pa = 0.0f;
    float pr = 0.0f;
    float pg = 0.0f;
    float pb = 0.0f;

    // Iterate over the surrounding 4x4 block of pixels
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            Cuda::vec2 mn(m, n);
            Cuda::vec2 xyi(gi + mn);

            xyi = Cuda::clamp(xyi, Cuda::vec2(0,0), in_size - Cuda::vec2(1,1));

            unsigned int pixel = in_pixels[(int)xyi.y * (int)in_size.x + (int)xyi.x];
            float weight = bicubic_weight(delta.x - m) * bicubic_weight(delta.y - n);

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

    out_pixels[(int)xy.y * (int)out_size.x + (int)xy.x] = d_argb(ia, ir, ig, ib);
}

extern "C" int cuda_bicubic_scale(const unsigned int* input_pixels, const Cuda::vec2& input_size, unsigned int* output_pixels, const Cuda::vec2& output_size) {
    unsigned int* d_input = nullptr;
    unsigned int* d_output = nullptr;

    size_t in_size = input_size.x * input_size.y * sizeof(unsigned int);
    size_t out_size = output_size.x * output_size.y * sizeof(unsigned int);

    cudaMalloc((void**)&d_input, in_size);
    cudaMemcpy(d_input, input_pixels, in_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_output, out_size);

    const Cuda::vec2 ratio = input_size / output_size;

    int numPixels = output_size.x * output_size.y;
    int blockSize = 256;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;
    bicubic_scale_kernel<<<numBlocks, blockSize>>>(
        d_input, input_size,
        d_output, output_size,
        ratio);
    cudaDeviceSynchronize();

    cudaMemcpy(output_pixels, d_output, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0; // success
}

__global__ void overlay_kernel(
    unsigned int* background, const Cuda::vec2 b_size,
    unsigned int* foreground, const Cuda::vec2 f_size,
    const Cuda::vec2 delta, const float opacity)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= f_size.x * f_size.y) return;

    int x = idx % (int)f_size.x;
    int y = idx / f_size.x;

    d_atomic_overlay_pixel(delta + Cuda::vec2(x, y), foreground[y * (int)f_size.x + x], opacity, background, b_size);
}

__global__ void overlay_rotation_kernel(
    unsigned int* background, const Cuda::vec2 b_size,
    unsigned int* foreground, const Cuda::vec2 f_size,
    const Cuda::vec2 delta, const float opacity, const float angle_rad)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= b_size.x * b_size.y) return;

    const Cuda::vec2 b_pixel(idx % (int)b_size.x, idx / b_size.x);

    // Compute position relative to overlay top-left
    const Cuda::vec2 relative(b_pixel - delta);

    // Center of the foreground
    const Cuda::vec2 f_center = (f_size - Cuda::vec2(1,1)) * 0.5f;

    // Translate to center, apply inverse rotation, translate back
    const Cuda::vec2 centered = relative - f_center;
    const Cuda::vec2 rotated = Cuda::cis(angle_rad);
    // inverse rotation by -angle -> use cos, -sin
    float srcx =  rotated.x * centered.x + rotated.y * centered.y + f_center.x;
    float srcy = -rotated.y * centered.x + rotated.x * centered.y + f_center.y;

    if (srcx < 0.0f || srcx >= static_cast<float>(f_size.x - 1) ||
        srcy < 0.0f || srcy >= static_cast<float>(f_size.y - 1)) {
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

    unsigned int p00 = foreground[y0 * (int)f_size.x + x0];
    unsigned int p10 = foreground[y0 * (int)f_size.x + x1];
    unsigned int p01 = foreground[y1 * (int)f_size.x + x0];
    unsigned int p11 = foreground[y1 * (int)f_size.x + x1];

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

    d_atomic_overlay_pixel(b_pixel, d_argb(255, rf, gf, bf), fg_alpha, background, b_size);
}

extern "C" void cuda_overlay (
    unsigned int* h_background, const Cuda::vec2& b_size,
    unsigned int* h_foreground, const Cuda::vec2& f_size,
    const Cuda::vec2& delta, const float opacity, const float angle_rad)
{
    if (opacity == 0.0f) return;

    unsigned int* d_background = nullptr;
    unsigned int* d_foreground = nullptr;

    size_t bg_size = b_size.x * b_size.y * sizeof(unsigned int);
    size_t fg_size = f_size.x * f_size.y * sizeof(unsigned int);

    cudaMalloc((void**)&d_background, bg_size);
    cudaMemcpy(d_background, h_background, bg_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_foreground, fg_size);
    cudaMemcpy(d_foreground, h_foreground, fg_size, cudaMemcpyHostToDevice);

    if(angle_rad != 0) {
        // TODO instead use the envelope surrounding the rotation INTERSECT the background itself
        int numPixels = b_size.x * b_size.y;
        int blockSize = 256;
        int numBlocks = (numPixels + blockSize - 1) / blockSize;
        overlay_rotation_kernel<<<numBlocks, blockSize>>>(
            d_background, b_size,
            d_foreground, f_size,
            delta, opacity, angle_rad);
    } else {
        int numPixels = f_size.x * f_size.y;
        int blockSize = 256;
        int numBlocks = (numPixels + blockSize - 1) / blockSize;
        overlay_kernel<<<numBlocks, blockSize>>>(
            d_background, b_size,
            d_foreground, f_size,
            delta, opacity);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_background, d_background, bg_size, cudaMemcpyDeviceToHost);

    cudaFree(d_background);
    cudaFree(d_foreground);
}
