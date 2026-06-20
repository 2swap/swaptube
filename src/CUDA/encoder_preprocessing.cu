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
    int uv_x = (blockIdx.x * blockDim.x + threadIdx.x);
    int uv_y = (blockIdx.y * blockDim.y + threadIdx.y);
    
    int x = uv_x * 2;
    int y = uv_y * 2;

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
            pixel = Cuda::color_combine(bg, pixel);

            uint8_t r = Cuda::getr(pixel);
            uint8_t g = Cuda::getg(pixel);
            uint8_t b = Cuda::getb(pixel);

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

    int idx = uv_y * uv_pitch + uv_x * 2;

    uv_plane[idx + 0] = u10;
    uv_plane[idx + 1] = v10;
}

// Reference VA-API codec YUV planes as device pointers, like CUDA does automatically
__host__ void createVaapiYuvPlanes(
    uint16_t** d_y_plane,
    uint16_t** d_uv_plane,
    int width, int height,
    int& y_pitch, int& uv_pitch, 
    int file_descriptor, size_t mem_size, 
    unsigned long long y_offset, unsigned long long uv_offset)
{
    cudaExternalMemory_t ext_mem;
    cudaExternalMemoryHandleDesc mem_handle_desc = {};

    mem_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    mem_handle_desc.handle.fd = file_descriptor;
    mem_handle_desc.size = mem_size;

    if(cudaImportExternalMemory(&ext_mem, &mem_handle_desc) != cudaSuccess) {
        throw runtime_error("Failed to import DMA-BUF into HIP");
    }

    cudaExternalMemoryBufferDesc buf_desc_y = {};
    buf_desc_y.offset = y_offset;
    buf_desc_y.size = y_pitch * height * sizeof(uint16_t);

    void* d_y_plane_new = nullptr;
    cudaExternalMemoryGetMappedBuffer(&d_y_plane_new, ext_mem, &buf_desc_y);

    cudaExternalMemoryBufferDesc buf_desc_uv = {};
    buf_desc_uv.offset = uv_offset;
    buf_desc_uv.size = uv_pitch * (height / 2) * sizeof(uint16_t);

    void* d_uv_plane_new = nullptr;
    cudaExternalMemoryGetMappedBuffer(&d_uv_plane_new, ext_mem, &buf_desc_uv);

    *d_y_plane = reinterpret_cast<uint16_t*>(d_y_plane_new);
    *d_uv_plane = reinterpret_cast<uint16_t*>(d_uv_plane_new);
    
    cudaDestroyExternalMemory(ext_mem);
}

extern "C" void preprocess_argb_to_p010(
    const uint32_t* d_argb,
    uint16_t* d_y_plane,
    uint16_t* d_uv_plane,
    int fd,
    size_t obj_size,
    int width,
    int height,
    int y_pitch_bytes,
    int uv_pitch_bytes,
    unsigned long long y_offset,
    unsigned long long uv_offset,
    uint32_t bg)
{
    int y_pitch = y_pitch_bytes / sizeof(uint16_t);
    int uv_pitch = uv_pitch_bytes / sizeof(uint16_t);

    #ifdef USE_AMD
    createVaapiYuvPlanes(
        &d_y_plane, &d_uv_plane, 
        width, height, 
        y_pitch, uv_pitch, 
        fd, 
        obj_size, 
        y_offset, uv_offset);
    #endif

    dim3 block(16, 16);
    dim3 grid((width / 2 + 15) / 16, (height / 2 + 15) / 16);
    argb_to_p010<<<grid, block>>>(
        d_argb, 
        d_y_plane, d_uv_plane, 
        width, height, 
        y_pitch, 
        uv_pitch, 
        bg);
    cudaDeviceSynchronize();
}
