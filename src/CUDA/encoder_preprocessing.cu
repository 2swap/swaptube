#include <cuda_runtime.h>
#include <stdint.h>
#include "color.cuh"

__device__ __forceinline__ uint16_t clamp10(float v)
{
    v = fminf(fmaxf(v, 0.0f), 1023.0f);
    return (uint16_t)v;
}

// BT.709 conversion (good default for NVENC pipelines)
__device__ __forceinline__ void rgb_to_yuv(
    uint8_t r, uint8_t g, uint8_t b,
    float &y, float &u, float &v)
{
    float rf = r;
    float gf = g;
    float bf = b;

    y =  0.2126f * rf + 0.7152f * gf + 0.0722f * bf;
    u = -0.1146f * rf - 0.3854f * gf + 0.5000f * bf;
    v =  0.5000f * rf - 0.4542f * gf - 0.0458f * bf;
}

__global__ void argb_to_p010(
    const uint32_t* __restrict__ argb,
    uint16_t* __restrict__ y_plane,
    uint16_t* __restrict__ uv_plane,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    uint32_t bg)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    if (x >= width || y >= height) return;

    float u_sum = 0.0f;
    float v_sum = 0.0f;

    #pragma unroll
    for (int dy = 0; dy < 2; dy++)
    {
        #pragma unroll
        for (int dx = 0; dx < 2; dx++)
        {
            int ix = x + dx;
            int iy = y + dy;

            if (ix >= width || iy >= height) continue;

            uint32_t pixel = argb[iy * width + ix];
            pixel = d_color_combine(bg, pixel);

            uint8_t r = d_getr(pixel);
            uint8_t g = d_getg(pixel);
            uint8_t b = d_getb(pixel);

            float yf, uf, vf;
            rgb_to_yuv(r, g, b, yf, uf, vf);

            float y10f = 64.0f + (yf / 255.0f * 876.0f);
            uint16_t y10 = clamp10(y10f) << 6;

            y_plane[iy * y_pitch + ix] = y10;

            u_sum += uf;
            v_sum += vf;
        }
    }

    u_sum *= 0.25f;
    v_sum *= 0.25f;

    // BT.709 limited range: Cb/Cr 64-960 in 10-bit, left-aligned in 16-bit.
    // rgb_to_yuv produces chroma in [-127.5,127.5], so we need to map that to [64,960].
    float u10f = 512.0f + (u_sum * (896.0f / 255.0f));
    float v10f = 512.0f + (v_sum * (896.0f / 255.0f));

    uint16_t u10 = clamp10(u10f) << 6;
    uint16_t v10 = clamp10(v10f) << 6;

    int uv_x = x / 2;
    int uv_y = y / 2;

    int idx = uv_y * uv_pitch + uv_x * 2;

    uv_plane[idx + 0] = u10;
    uv_plane[idx + 1] = v10;
}

extern "C" void preprocess_argb_to_p010(
    const uint32_t* d_argb,
    uint16_t* d_y_plane,
    uint16_t* d_uv_plane,
    int width,
    int height,
    int y_stride,
    int uv_stride,
    uint32_t bg)
{
    int y_pitch = y_stride / sizeof(uint16_t);
    int uv_pitch = uv_stride / sizeof(uint16_t);
    dim3 block(16, 16);
    dim3 grid((width / 2 + 15) / 16, (height / 2 + 15) / 16);

    argb_to_p010<<<grid, block>>>(d_argb, d_y_plane, d_uv_plane, width, height, y_pitch, uv_pitch, bg);
    cudaDeviceSynchronize();
}
