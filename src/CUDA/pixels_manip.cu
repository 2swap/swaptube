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
    const unsigned int* in_pixels, const int in_w, const int in_h,
    unsigned int* out_pixels, const int out_w, const int out_h,
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

            unsigned int pixel = in_pixels[yi * in_w + xi];
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

extern "C" int cuda_bicubic_scale(const unsigned int* input_pixels, int input_w, int input_h, unsigned int* output_pixels, int output_w, int output_h) {
    unsigned int* d_input = nullptr;
    unsigned int* d_output = nullptr;

    size_t in_size = input_w * input_h * sizeof(unsigned int);
    size_t out_size = output_w * output_h * sizeof(unsigned int);

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
    unsigned int* background, const int bw, const int bh,
    unsigned int* foreground, const int fw, const int fh,
    const int dx, const int dy, const float opacity)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= fw * fh) return;

    int x = idx % fw;
    int y = idx / fw;

    d_overlay_pixel(x+dx, y+dy, foreground[y * fw + x], opacity, background, bw, bh);
}

__global__ void overlay_rotation_kernel(
    unsigned int* background, const int bw, const int bh,
    unsigned int* foreground, const int fw, const int fh,
    const int dx, const int dy, const float opacity, const float angle_rad)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= bw * bh) return;

    int bx = idx % bw;
    int by = idx / bw;

    // Compute position relative to overlay top-left
    float lx = static_cast<float>(bx - dx);
    float ly = static_cast<float>(by - dy);

    // Center of the foreground
    float cx = (fw - 1) * 0.5f;
    float cy = (fh - 1) * 0.5f;

    // Translate to center, apply inverse rotation, translate back
    float fx = lx - cx;
    float fy = ly - cy;
    float cosA = cosf(angle_rad);
    float sinA = sinf(angle_rad);
    // inverse rotation by -angle -> use cos, -sin
    float srcx =  cosA * fx + sinA * fy + cx;
    float srcy = -sinA * fx + cosA * fy + cy;

    if (srcx < 0.0f || srcx >= static_cast<float>(fw - 1) ||
        srcy < 0.0f || srcy >= static_cast<float>(fh - 1)) {
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

    unsigned int p00 = foreground[y0 * fw + x0];
    unsigned int p10 = foreground[y0 * fw + x1];
    unsigned int p01 = foreground[y1 * fw + x0];
    unsigned int p11 = foreground[y1 * fw + x1];

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

    d_overlay_pixel(bx, by, d_argb(255, rf, gf, bf), fg_alpha, background, bw, bh);
}

extern "C" void cuda_overlay (
    unsigned int* h_background, const vec2& b_size,
    unsigned int* h_foreground, const vec2& f_size,
    const vec2& delta, const float opacity, const float angle_rad)
{
    if (opacity == 0.0f) return;

    unsigned int* d_background = nullptr;
    unsigned int* d_foreground = nullptr;

    size_t bg_size = bw * bh * sizeof(unsigned int);
    size_t fg_size = fw * fh * sizeof(unsigned int);

    cudaMalloc((void**)&d_background, bg_size);
    cudaMemcpy(d_background, h_background, bg_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_foreground, fg_size);
    cudaMemcpy(d_foreground, h_foreground, fg_size, cudaMemcpyHostToDevice);

    if(angle_rad != 0) {
        // TODO instead use the envelope surrounding the rotation INTERSECT the background itself
        int numPixels = bw * bh; // iterate over full background and map into rotated foreground via inverse transform
        int blockSize = 256;
        int numBlocks = (numPixels + blockSize - 1) / blockSize;
        overlay_rotation_kernel<<<numBlocks, blockSize>>>(
            d_background, bw, bh,
            d_foreground, fw, fh,
            dx, dy, opacity, angle_rad);
    } else {
        
        int numPixels = fw * fh;
        int blockSize = 256;
        int numBlocks = (numPixels + blockSize - 1) / blockSize;
        overlay_kernel<<<numBlocks, blockSize>>>(
            d_background, bw, bh,
            d_foreground, fw, fh,
            dx, dy, opacity);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_background, d_background, bg_size, cudaMemcpyDeviceToHost);

    cudaFree(d_background);
    cudaFree(d_foreground);
}
